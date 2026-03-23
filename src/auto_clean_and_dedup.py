"""Automatically clean and deduplicate people records via LLM prompts.

Workflow:
1. Load people through ``PeopleStore``.
2. Clean name/affiliation/country in LLM batches (up to 20 people per query).
3. Collapse exact-name duplicates that may be introduced by cleaning.
4. Detect and merge likely duplicate names.
5. Persist via ``PeopleStore.overwrite_all`` (unless ``--dry-run``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response
from people import PeopleStore


CLEAN_BATCH_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "clean_people_batch.txt"
DUPLICATE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "find_duplicate_people.txt"
MERGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "merge_duplicate_people_limited.txt"

COUNTRY_CODE_RE = re.compile(r"^[A-Z]{3}$")
CLEANING_BATCH_SIZE = 20
DRY_RUN_LIMIT = 50
MERGE_RETRY_COUNT = 2


class CleanBatchPerson(BaseModel):
    original_name: str
    name: str
    affiliation: str
    country: str | None = None


class CleanBatchResult(BaseModel):
    people: list[CleanBatchPerson]


class DuplicateGroup(BaseModel):
    primary_name: str
    duplicate_names: list[str]
    rationale: str | None = None


class DuplicateDetectionResult(BaseModel):
    groups: list[DuplicateGroup]


class MergedMinimalPerson(BaseModel):
    name: str
    affiliation: str
    country: str | None = None


class MergeDuplicateResult(BaseModel):
    merged_person: MergedMinimalPerson


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and deduplicate people entries via LLM. "
            "Dry run processes only the first 50 entries and writes nothing."
        )
    )
    parser.add_argument(
        "--people-file",
        type=Path,
        default=None,
        help="Optional people JSONL path override. By default, people.py chooses the store location.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RESPONSES_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_RESPONSES_MODEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 50 records and do not write updates.",
    )
    return parser.parse_args(argv)


def _load_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {path}") from exc


def _normalize_country_code(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().upper()
    if not normalized:
        return None
    if not COUNTRY_CODE_RE.fullmatch(normalized):
        raise ValueError(f"Country code is not ISO-3166 alpha-3: {value!r}")
    return normalized


def _minimal_person_for_llm(person: dict[str, Any]) -> dict[str, Any]:
    country_value = person.get("country")
    country = country_value if isinstance(country_value, str) else None

    affiliation_value = person.get("affiliation")
    affiliation = affiliation_value if isinstance(affiliation_value, str) else None

    return {
        "name": str(person.get("name", "")).strip(),
        "affiliation": affiliation,
        "country": country,
    }


def _clean_batch_prompt(batch: list[dict[str, Any]]) -> str:
    template = _load_prompt(CLEAN_BATCH_PROMPT_PATH)
    payload = [_minimal_person_for_llm(person) for person in batch]
    return template.format(people=json.dumps(payload, ensure_ascii=False))


def _duplicate_prompt(names: list[str]) -> str:
    template = _load_prompt(DUPLICATE_PROMPT_PATH)
    sorted_names = sorted(names, key=str.casefold)
    return template.format(names=json.dumps(sorted_names, ensure_ascii=False))


def _merge_prompt(canonical_name: str, first: dict[str, Any], second: dict[str, Any]) -> str:
    template = _load_prompt(MERGE_PROMPT_PATH)
    payload = {
        "canonical_name": canonical_name,
        "record_a": _minimal_person_for_llm(first),
        "record_b": _minimal_person_for_llm(second),
    }
    return template.format(payload=json.dumps(payload, ensure_ascii=False))


def _merge_values_local(existing: Any, new_value: Any) -> Any:
    if isinstance(existing, dict) and isinstance(new_value, dict):
        return _merge_dict_local(existing, new_value)

    if isinstance(existing, list) and isinstance(new_value, list):
        merged = list(existing)
        seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged}
        for item in new_value:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged

    return new_value


def _merge_dict_local(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if key in merged:
            merged[key] = _merge_values_local(merged[key], value)
        else:
            merged[key] = value
    return merged


def _detail_score(record: dict[str, Any]) -> int:
    score = 0
    for value in record.values():
        if isinstance(value, str) and value.strip():
            score += 1
        elif isinstance(value, list) and value:
            score += len(value)
        elif isinstance(value, dict) and value:
            score += len(value)
        elif value is not None:
            score += 1
    return score


def _fallback_merge_records(
    canonical_name: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    preferred = second if _detail_score(second) >= _detail_score(first) else first
    merged = _merge_dict_local(first, second)

    for key in ("affiliation", "homepage", "email", "country", "gender"):
        preferred_value = preferred.get(key)
        if preferred_value is not None and not (isinstance(preferred_value, str) and not preferred_value.strip()):
            merged[key] = preferred_value

    merged["name"] = canonical_name
    return merged


def _apply_country(record: dict[str, Any], country_value: str | None) -> None:
    try:
        normalized_country = _normalize_country_code(country_value)
    except ValueError:
        normalized_country = None

    if normalized_country is not None:
        record["country"] = normalized_country
        return

    existing_country = record.get("country")
    if isinstance(existing_country, str):
        try:
            normalized_existing = _normalize_country_code(existing_country)
        except ValueError:
            normalized_existing = None
        if normalized_existing is not None:
            record["country"] = normalized_existing
            return

    record.pop("country", None)


def _clean_people_batch(batch: list[dict[str, Any]], start: int, total: int, model: str) -> list[dict[str, Any]]:
    print(f"[LLM] Cleaning people batch {start + 1}-{start + len(batch)} of {total}")

    parsed = parse_structured_response(
        input_text=_clean_batch_prompt(batch),
        response_model=CleanBatchResult,
        description="clean people batch",
        model=model,
    )

    parsed_by_original: dict[str, CleanBatchPerson] = {}
    for item in parsed.people:
        key = item.original_name.strip()
        if key and key not in parsed_by_original:
            parsed_by_original[key] = item

    cleaned_records: list[dict[str, Any]] = []
    for original in batch:
        original_name = str(original.get("name", "")).strip()
        item = parsed_by_original.get(original_name)
        if item is None:
            print(f"[Warn] Missing cleaned output for {original_name}; keeping original values")
            cleaned_records.append(dict(original))
            continue

        cleaned = dict(original)

        cleaned_name = item.name.strip()
        if cleaned_name:
            cleaned["name"] = cleaned_name

        cleaned_affiliation = item.affiliation.strip()
        if cleaned_affiliation:
            cleaned["affiliation"] = cleaned_affiliation

        _apply_country(cleaned, item.country)
        cleaned_records.append(cleaned)

    return cleaned_records


def _clean_people_batches(people: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    if not people:
        return []

    cleaned_records: list[dict[str, Any]] = []
    total = len(people)

    for start in range(0, total, CLEANING_BATCH_SIZE):
        batch = people[start : start + CLEANING_BATCH_SIZE]
        cleaned_records.extend(_clean_people_batch(batch, start, total, model))

    return cleaned_records


def _detect_duplicate_groups(names: list[str], model: str) -> list[DuplicateGroup]:
    print(f"[LLM] Detecting likely duplicate names across {len(names)} people")
    parsed = parse_structured_response(
        input_text=_duplicate_prompt(names),
        response_model=DuplicateDetectionResult,
        description="detect duplicate names",
        model=model,
    )
    return parsed.groups


def _normalize_detected_groups(
    groups: list[DuplicateGroup],
    people_by_name: dict[str, dict[str, Any]],
) -> list[DuplicateGroup]:
    normalized: list[DuplicateGroup] = []
    consumed_names: set[str] = set()

    for group in groups:
        primary_name = group.primary_name.strip()
        if primary_name not in people_by_name:
            print(f"[Skip] LLM primary_name not found: {primary_name}")
            continue

        seen_in_group: set[str] = {primary_name}
        filtered_duplicates: list[str] = []
        for duplicate_name in group.duplicate_names:
            candidate = duplicate_name.strip()
            if not candidate or candidate == primary_name:
                continue
            if candidate not in people_by_name:
                print(f"[Skip] LLM duplicate_name not found: {candidate}")
                continue
            if candidate in seen_in_group:
                continue
            seen_in_group.add(candidate)
            filtered_duplicates.append(candidate)

        if not filtered_duplicates:
            continue

        full_group_names = [primary_name, *filtered_duplicates]
        if any(name in consumed_names for name in full_group_names):
            print(f"[Skip] Overlapping duplicate group ignored: {full_group_names}")
            continue

        consumed_names.update(full_group_names)
        normalized.append(
            DuplicateGroup(
                primary_name=primary_name,
                duplicate_names=filtered_duplicates,
                rationale=group.rationale,
            )
        )

    return normalized


def _merge_pair(canonical_name: str, first: dict[str, Any], second: dict[str, Any], model: str) -> dict[str, Any]:
    print(f"[LLM] Merging into canonical entry: {canonical_name}")
    prompt = _merge_prompt(canonical_name, first, second)

    last_error: Exception | None = None
    for attempt in range(1, MERGE_RETRY_COUNT + 2):
        try:
            parsed = parse_structured_response(
                input_text=prompt,
                response_model=MergeDuplicateResult,
                description="merge duplicate pair",
                model=model,
            )
            merged = _merge_dict_local(first, second)
            merged["name"] = canonical_name

            affiliation = parsed.merged_person.affiliation.strip()
            if affiliation:
                merged["affiliation"] = affiliation

            _apply_country(merged, parsed.merged_person.country)
            return merged
        except ValueError as exc:
            last_error = exc
            if attempt <= MERGE_RETRY_COUNT:
                print(
                    f"[Warn] Invalid merge output for {canonical_name} "
                    f"(attempt {attempt}/{MERGE_RETRY_COUNT + 1}); retrying"
                )

    print(
        f"[Warn] Falling back to local merge for {canonical_name} after malformed model output: {last_error}"
    )
    return _fallback_merge_records(canonical_name, first, second)


def _collapse_exact_name_duplicates(
    records: list[dict[str, Any]],
    model: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    by_name: dict[str, dict[str, Any]] = {}
    merged_count = 0

    for record in records:
        name = str(record.get("name", "")).strip()
        if not name:
            continue

        normalized = dict(record)
        normalized["name"] = name

        if name in by_name:
            by_name[name] = _merge_pair(name, by_name[name], normalized, model)
            merged_count += 1
        else:
            by_name[name] = normalized

    return by_name, merged_count


def _merge_groups(
    groups: list[DuplicateGroup],
    people_by_name: dict[str, dict[str, Any]],
    model: str,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    merged_updates: dict[str, dict[str, Any]] = {}
    removed_names: set[str] = set()

    for group in groups:
        canonical_name = group.primary_name
        merged_record = dict(people_by_name[canonical_name])

        for duplicate_name in group.duplicate_names:
            merged_record = _merge_pair(
                canonical_name,
                merged_record,
                people_by_name[duplicate_name],
                model,
            )
            removed_names.add(duplicate_name)

        merged_record["name"] = canonical_name
        merged_updates[canonical_name] = merged_record

    return merged_updates, removed_names


def _plan_groups(groups: list[DuplicateGroup]) -> None:
    print(f"[Plan] Applying {len(groups)} duplicate merge groups")
    for group in groups:
        print(
            json.dumps(
                {
                    "primary_name": group.primary_name,
                    "duplicate_names": group.duplicate_names,
                    "rationale": group.rationale,
                },
                ensure_ascii=False,
            )
        )


def run_auto_clean_and_dedup(
    store: PeopleStore,
    model: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    loaded_people = store.load()
    if not loaded_people:
        print("[Info] People store is empty")
        return 0, 0, 0

    all_names = sorted(loaded_people, key=str.casefold)
    selected_names = all_names[:DRY_RUN_LIMIT] if dry_run else all_names
    selected_records = [dict(loaded_people[name]) for name in selected_names]

    if dry_run:
        print(f"[Dry Run] Processing {len(selected_records)} of {len(all_names)} total entries")
        cleaned_records = _clean_people_batches(selected_records, model)
        cleaned_by_name, exact_name_merged = _collapse_exact_name_duplicates(cleaned_records, model)
    else:
        print(f"[Apply] Processing {len(selected_records)} entries")
        total = len(selected_records)
        remaining_original_by_name = {name: dict(loaded_people[name]) for name in selected_names}
        processed_cleaned_by_name: dict[str, dict[str, Any]] = {}
        exact_name_merged = 0

        for start in range(0, total, CLEANING_BATCH_SIZE):
            batch = selected_records[start : start + CLEANING_BATCH_SIZE]
            cleaned_batch = _clean_people_batch(batch, start, total, model)

            for original in batch:
                original_name = str(original.get("name", "")).strip()
                if original_name:
                    remaining_original_by_name.pop(original_name, None)

            for cleaned in cleaned_batch:
                cleaned_name = str(cleaned.get("name", "")).strip()
                if not cleaned_name:
                    continue

                normalized = dict(cleaned)
                normalized["name"] = cleaned_name

                if cleaned_name in processed_cleaned_by_name:
                    processed_cleaned_by_name[cleaned_name] = _merge_pair(
                        cleaned_name,
                        processed_cleaned_by_name[cleaned_name],
                        normalized,
                        model,
                    )
                    exact_name_merged += 1
                else:
                    processed_cleaned_by_name[cleaned_name] = normalized

            checkpoint_people = dict(remaining_original_by_name)
            for name, record in processed_cleaned_by_name.items():
                if name in checkpoint_people:
                    checkpoint_people[name] = _fallback_merge_records(name, checkpoint_people[name], record)
                else:
                    checkpoint_people[name] = record

            written = store.overwrite_all(checkpoint_people.values())
            print(
                f"[Store] Checkpoint saved after cleaning batch "
                f"{start + 1}-{start + len(batch)} ({written} records)"
            )

        cleaned_by_name = dict(processed_cleaned_by_name)

    names_after_cleaning = sorted(cleaned_by_name, key=str.casefold)
    if not names_after_cleaning:
        print("[Info] No valid cleaned records to process")
        return 0, 0, 0

    proposed_groups = _detect_duplicate_groups(names_after_cleaning, model)
    groups = _normalize_detected_groups(proposed_groups, cleaned_by_name)

    if groups:
        _plan_groups(groups)
        merged_updates, removed_names = _merge_groups(groups, cleaned_by_name, model)
        for removed_name in removed_names:
            cleaned_by_name.pop(removed_name, None)
        for merged_name, merged_record in merged_updates.items():
            cleaned_by_name[merged_name] = merged_record
    else:
        removed_names = set()
        print("[Info] No duplicate groups identified after cleaning")

    merged_duplicates = exact_name_merged + len(removed_names)

    if dry_run:
        print(
            "[Dry Run] Final record count for sample: "
            f"{len(cleaned_by_name)} (merged duplicates: {merged_duplicates})"
        )
        return len(cleaned_by_name), merged_duplicates, len(selected_records)

    written = store.overwrite_all(cleaned_by_name.values())
    print(f"[Store] Wrote {written} records")
    return written, merged_duplicates, len(selected_records)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = PeopleStore(path=args.people_file) if args.people_file is not None else PeopleStore()
    final_count, merged_count, processed_count = run_auto_clean_and_dedup(
        store=store,
        model=args.model,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "processed_entries": processed_count,
                "final_count": final_count,
                "merged_duplicates": merged_count,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
