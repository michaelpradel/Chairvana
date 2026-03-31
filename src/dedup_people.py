"""Find and merge likely duplicate people records via LLM prompts.

Workflow:
1. Load all people via ``DataStore``.
2. Ask an LLM to identify high-confidence duplicate name groups.
3. For each duplicate group, ask an LLM to merge records pairwise.
4. Persist the full merged snapshot back via ``DataStore.overwrite_all``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response
from data_store import DataStore


DUPLICATE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "find_duplicate_people.txt"
MERGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "merge_duplicate_people.txt"
CHECKPOINT_GROUP_SIZE = 10
MERGE_RETRY_COUNT = 2


class DuplicateGroup(BaseModel):
    primary_name: str
    duplicate_names: list[str]
    rationale: str | None = None


class DuplicateDetectionResult(BaseModel):
    groups: list[DuplicateGroup]


class MergeDuplicateResult(BaseModel):
    merged_person_json: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find and merge duplicate people entries via LLM.")
    parser.add_argument(
        "--people-file",
        type=Path,
        default=None,
        help="Optional people JSONL path override. By default, data_store.py chooses the store location.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RESPONSES_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_RESPONSES_MODEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report planned duplicate merges without writing updates.",
    )
    parser.add_argument(
        "--dry-run-max-groups",
        type=int,
        default=5,
        help="Maximum duplicate groups to merge during --dry-run (default: 5). Use 0 or negative for no limit.",
    )
    return parser.parse_args(argv)


def _load_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {path}") from exc


def _duplicate_prompt(names: list[str]) -> str:
    template = _load_prompt(DUPLICATE_PROMPT_PATH)
    sorted_names = sorted(names, key=str.casefold)
    return template.format(names=json.dumps(sorted_names, ensure_ascii=False))


def _merge_prompt(canonical_name: str, first: dict[str, Any], second: dict[str, Any]) -> str:
    template = _load_prompt(MERGE_PROMPT_PATH)
    payload = {
        "canonical_name": canonical_name,
        "record_a": first,
        "record_b": second,
    }
    return template.format(payload=json.dumps(payload, ensure_ascii=False))


def _parse_merged_person_json(raw: str, canonical_name: str) -> dict[str, Any]:
    candidate = raw.strip()

    # Some model outputs still wrap JSON in markdown fences despite instructions.
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            snippet = candidate[:300].replace("\n", "\\n")
            raise ValueError(
                f"Could not parse merged_person_json for {canonical_name}: {exc}. "
                f"Response snippet: {snippet}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"merged_person_json for {canonical_name} must decode to an object")

    return parsed


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

    # Keep selected scalar values from the more detailed record when both are present.
    for key in ("affiliation", "homepage", "email", "country", "gender"):
        preferred_value = preferred.get(key)
        if preferred_value is not None and not (isinstance(preferred_value, str) and not preferred_value.strip()):
            merged[key] = preferred_value

    merged["name"] = canonical_name
    return merged


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
            merged = _parse_merged_person_json(parsed.merged_person_json, canonical_name)
            merged["name"] = canonical_name
            return merged
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt <= MERGE_RETRY_COUNT:
                print(
                    f"[Warn] Invalid merge JSON for {canonical_name} "
                    f"(attempt {attempt}/{MERGE_RETRY_COUNT + 1}); retrying"
                )

    print(
        f"[Warn] Falling back to local merge for {canonical_name} after malformed model output: {last_error}"
    )
    return _fallback_merge_records(canonical_name, first, second)


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


def dedup_people(
    store: DataStore,
    model: str,
    dry_run: bool,
    dry_run_max_groups: int,
) -> tuple[int, int]:
    people_by_name = store.load()
    all_names = sorted(people_by_name, key=str.casefold)
    if not all_names:
        print("[Info] People store is empty")
        return 0, 0

    proposed_groups = _detect_duplicate_groups(all_names, model)
    groups = _normalize_detected_groups(proposed_groups, people_by_name)

    if not groups:
        print("[Info] No duplicate groups identified")
        return len(all_names), 0

    if dry_run and dry_run_max_groups > 0 and len(groups) > dry_run_max_groups:
        print(f"[Dry Run] Limiting merge preview to first {dry_run_max_groups} groups out of {len(groups)}")
        groups = groups[:dry_run_max_groups]

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

    if dry_run:
        merged_updates, removed_names = _merge_groups(groups, people_by_name, model)

        final_people = dict(people_by_name)
        for removed_name in removed_names:
            final_people.pop(removed_name, None)
        for merged_name, merged_record in merged_updates.items():
            final_people[merged_name] = merged_record

        merged_count = len(removed_names)
        print(f"\n[Merged Entries]")
        for merged_name in sorted(merged_updates, key=str.casefold):
            merged_record = merged_updates[merged_name]
            print(json.dumps(merged_record, ensure_ascii=False, indent=2))
        print(
            f"\n[Dry Run] Would merge away {merged_count} duplicate entries; "
            f"final record count would be {len(final_people)}"
        )
        return len(final_people), merged_count

    current_people = dict(people_by_name)
    merged_count_total = 0
    total_groups = len(groups)
    processed_groups = 0

    for start_index in range(0, total_groups, CHECKPOINT_GROUP_SIZE):
        batch = groups[start_index : start_index + CHECKPOINT_GROUP_SIZE]
        print(
            f"[Apply] Resolving groups {start_index + 1}-{start_index + len(batch)} "
            f"of {total_groups}"
        )

        merged_updates, removed_names = _merge_groups(batch, current_people, model)
        for removed_name in removed_names:
            current_people.pop(removed_name, None)
        for merged_name, merged_record in merged_updates.items():
            current_people[merged_name] = merged_record

        merged_count_total += len(removed_names)
        processed_groups += len(batch)

        written = store.overwrite_all(current_people.values())
        print(
            f"[Store] Checkpoint saved after {processed_groups}/{total_groups} groups "
            f"({written} entries, merged duplicates so far: {merged_count_total})"
        )

    return len(current_people), merged_count_total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = DataStore(path=args.people_file) if args.people_file is not None else DataStore()
    final_count, merged_count = dedup_people(
        store,
        args.model,
        args.dry_run,
        args.dry_run_max_groups,
    )
    print(json.dumps({"final_count": final_count, "merged_duplicates": merged_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
