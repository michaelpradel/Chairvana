"""Determine and store a person's gender in the people store.

Two-tier approach:
1. Ask an LLM to classify gender from first name only as:
   - very_likely_male
   - very_likely_female
   - unclear
2. If tier 1 is unclear, ask an LLM to inspect the person's homepage and
   decide male or female. If homepage is missing, first discover homepage/email
   with ``find_homepage_and_email`` and update the person record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response
from data_store import DataStore
from web_search import find_homepage_and_email


TIER1_NAME_GENDER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_gender_tier1_name.txt"
)
TIER1_BATCH_NAME_GENDER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_gender_tier1_names_batch.txt"
)
TIER2_HOMEPAGE_GENDER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_gender_tier2_homepage.txt"
)
GENDER_SAVE_INTERVAL = 10


class Tier1NameGenderResult(BaseModel):
    classification: Literal["very_likely_male", "very_likely_female", "unclear"]


class GenderAssignment(BaseModel):
    name: str
    classification: Literal["very_likely_male", "very_likely_female", "unclear"]


class Tier1BatchNameGenderResult(BaseModel):
    """Batch result with list of name-to-gender assignments."""
    assignments: list[GenderAssignment]


class Tier2HomepageGenderResult(BaseModel):
    gender: Literal["male", "female"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer and store person gender (male/female) via a two-tier LLM workflow."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional person name to process (case-insensitive match). If omitted, process from the people store.",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=10,
        help="When processing from the people store, process only the first N people (default: 10).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="When processing from the people store, process all people (ignores --first-n).",
    )
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
    args = parser.parse_args(argv)
    if args.first_n <= 0:
        parser.error("--first-n must be a positive integer")
    return args


def _get_person_by_name(people: dict[str, dict[str, Any]], requested_name: str) -> tuple[str, dict[str, Any]]:
    if requested_name in people:
        return requested_name, people[requested_name]

    folded_requested = requested_name.casefold().strip()
    matches = [name for name in people if name.casefold() == folded_requested]
    if len(matches) == 1:
        matched_name = matches[0]
        return matched_name, people[matched_name]
    if not matches:
        raise ValueError(f"No person found with name: {requested_name}")
    raise ValueError(f"Ambiguous name {requested_name!r}: {matches}")


def _tier1_prompt(name: str) -> str:
    try:
        template = TIER1_NAME_GENDER_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {TIER1_NAME_GENDER_PROMPT_PATH}") from exc
    return template.format(name=name)


def _tier1_batch_prompt(names: list[str]) -> str:
    try:
        template = TIER1_BATCH_NAME_GENDER_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {TIER1_BATCH_NAME_GENDER_PROMPT_PATH}") from exc
    names_list = "\n".join(f"- {name}" for name in names)
    return template.format(names_list=names_list)


def _tier2_prompt(name: str, homepage: str) -> str:
    try:
        template = TIER2_HOMEPAGE_GENDER_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {TIER2_HOMEPAGE_GENDER_PROMPT_PATH}") from exc
    return template.format(name=name, homepage=homepage)


def infer_gender_tier1(name: str, model: str) -> str | None:
    print(f"[LLM][Tier 1] Querying name-only gender for: {name}")
    parsed = parse_structured_response(
        input_text=_tier1_prompt(name),
        response_model=Tier1NameGenderResult,
        description="infer gender by name (tier 1)",
        model=model,
    )
    print(f"[LLM][Tier 1] Classification for {name}: {parsed.classification}")

    if parsed.classification == "very_likely_male":
        return "male"
    if parsed.classification == "very_likely_female":
        return "female"
    return None


def infer_gender_tier1_batch(names: list[str], model: str) -> dict[str, str | None]:
    """Query tier1 gender for up to 50 names in a single batch request.
    
    Args:
        names: List of names to classify (max 50)
        model: OpenAI model to use
        
    Returns:
        dict mapping each name to gender ("male", "female") or None if "unclear"
    """
    if not names:
        return {}
    
    if len(names) > 50:
        raise ValueError(f"Batch size exceeds 50: {len(names)}")
    
    print(f"[LLM][Tier 1 Batch] Querying name-only gender for {len(names)} names")
    parsed = parse_structured_response(
        input_text=_tier1_batch_prompt(names),
        response_model=Tier1BatchNameGenderResult,
        description="infer gender batch by name (tier 1)",
        model=model,
    )
    
    result = {}
    for assignment in parsed.assignments:
        name = assignment.name
        classification = assignment.classification
        print(f"[LLM][Tier 1 Batch] Classification for {name}: {classification}")
        if classification == "very_likely_male":
            result[name] = "male"
        elif classification == "very_likely_female":
            result[name] = "female"
        else:
            result[name] = None
    
    return result


def infer_gender_from_homepage(name: str, homepage: str, model: str) -> str:
    print(f"[LLM][Tier 2] Querying homepage-based gender for: {name} ({homepage})")
    parsed = parse_structured_response(
        input_text=_tier2_prompt(name, homepage),
        response_model=Tier2HomepageGenderResult,
        description="infer gender from homepage (tier 2)",
        model=model,
        tools=[{"type": "web_search"}],
    )
    print(f"[LLM][Tier 2] Gender for {name}: {parsed.gender}")
    return parsed.gender


def _ensure_homepage(person: dict[str, Any], store: DataStore) -> dict[str, Any]:
    homepage = person.get("homepage")
    if isinstance(homepage, str) and homepage.strip():
        return person

    name = person.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Person record has no valid name")

    print(f"[Web] No homepage for {name}; querying homepage/email")
    resolved = find_homepage_and_email(name)
    updates: dict[str, str] = {}
    if resolved.homepage.strip():
        updates["homepage"] = resolved.homepage
    if resolved.email.strip():
        updates["email"] = resolved.email

    if not updates:
        print(f"[Web][Warn] Could not resolve homepage/email for {name}")
        return person

    updated = store.update_person(name, updates)
    saved_fields = "/".join(updates.keys())
    print(f"[Store] Saved {saved_fields} for {name}")
    return updated


def infer_gender_for_person(
    store: DataStore,
    person: dict[str, Any],
    model: str,
) -> tuple[str, str, bool]:
    name = person.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Person record must contain a non-empty 'name'")
    canonical_name = name.strip()

    tier1_gender = infer_gender_tier1(canonical_name, model)
    if tier1_gender is not None:
        return canonical_name, tier1_gender, False

    person_with_homepage = _ensure_homepage(person, store)
    homepage = person_with_homepage.get("homepage")
    if not isinstance(homepage, str) or not homepage.strip():
        raise ValueError(f"Could not obtain homepage for person: {canonical_name}")

    tier2_gender = infer_gender_from_homepage(canonical_name, homepage, model)
    return str(person_with_homepage["name"]), tier2_gender, True


def infer_and_store_gender_for_person(
    store: DataStore,
    person: dict[str, Any],
    model: str,
) -> tuple[str, str, bool]:
    name, gender, used_tier2 = infer_gender_for_person(store, person, model)
    updated = store.update_person(name, {"gender": gender})
    print(f"[Store] Saved gender for {updated['name']}: {gender}")
    return updated["name"], gender, used_tier2


def _flush_gender_updates(store: DataStore, pending_updates: list[dict[str, str]]) -> int:
    if not pending_updates:
        return 0

    _, updated_count = store.update_many(pending_updates)
    print(f"[Store] Checkpointed {updated_count} gender updates")
    pending_updates.clear()
    return updated_count


def infer_and_store_gender(
    store: DataStore,
    requested_name: str,
    model: str,
) -> tuple[str, str | None, bool, bool]:
    people = store.load()
    _, person = _get_person_by_name(people, requested_name)

    existing_gender = person.get("gender")
    if isinstance(existing_gender, str) and existing_gender.strip() in {"male", "female"}:
        normalized_existing_gender = existing_gender.strip()
        print(f"[Skip] Gender already set for {person['name']}: {normalized_existing_gender}")
        return person["name"], normalized_existing_gender, False, True

    name, gender, used_tier2 = infer_and_store_gender_for_person(store, person, model)
    return name, gender, used_tier2, False


def process_people_batch(store: DataStore, model: str, first_n: int, process_all: bool) -> int:
    people = store.list_people()
    selected = people if process_all else people[:first_n]
    print(f"[Batch] Processing {len(selected)} people")

    processed_count = 0
    skipped_count = 0
    pending_gender_updates: list[dict[str, str]] = []
    
    # Separate people into those needing processing and those to skip
    people_to_process: list[dict[str, Any]] = []
    for person in selected:
        person_name = str(person.get("name", "<unknown>"))
        existing_gender = person.get("gender")
        if isinstance(existing_gender, str) and existing_gender.strip() in {"male", "female"}:
            print(f"[Skip] Gender already set for {person_name}: {existing_gender.strip()}")
            skipped_count += 1
            continue
        people_to_process.append(person)
    
    # Process in batches of up to 50 for tier1
    batch_size = 50
    for batch_start in range(0, len(people_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(people_to_process))
        batch = people_to_process[batch_start:batch_end]
        batch_names = [str(p.get("name", "<unknown>")) for p in batch]
        
        print(f"[Batch] Processing names {batch_start + 1} to {batch_end} (batch size: {len(batch_names)})")
        
        # Run tier1 batch query
        tier1_results = infer_gender_tier1_batch(batch_names, model)
        
        # Collect tier2-needed people and update tier1-resolved people
        tier2_people: list[tuple[dict[str, Any], str]] = []  # (person, name)
        for person in batch:
            person_name = str(person.get("name", "<unknown>"))
            gender = tier1_results.get(person_name)
            
            if gender is not None:
                # Tier1 resolved the gender
                pending_gender_updates.append({"name": person_name, "gender": gender})
                processed_count += 1
            else:
                # Need tier2 (homepage-based)
                tier2_people.append((person, person_name))
        
        # Process tier2 for this batch's unclear results
        for person, person_name in tier2_people:
            try:
                person_with_homepage = _ensure_homepage(person, store)
                homepage = person_with_homepage.get("homepage")
                if not isinstance(homepage, str) or not homepage.strip():
                    print(f"[Skip] Could not obtain homepage for person: {person_name}")
                    skipped_count += 1
                    continue

                tier2_gender = infer_gender_from_homepage(person_name, homepage, model)
                pending_gender_updates.append({"name": person_name, "gender": tier2_gender})
                processed_count += 1
            except Exception as exc:
                print(f"[Skip] Tier2 failed for {person_name}: {exc}")
                skipped_count += 1
        
        # Checkpoint after processing this batch
        if len(pending_gender_updates) >= GENDER_SAVE_INTERVAL:
            _flush_gender_updates(store, pending_gender_updates)
    
    # Final checkpoint
    _flush_gender_updates(store, pending_gender_updates)

    print(f"[Batch] Completed. Updated: {processed_count}, skipped: {skipped_count}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = DataStore(path=args.people_file) if args.people_file is not None else DataStore()

    if args.name:
        name, gender, used_tier2, skipped = infer_and_store_gender(store, args.name, args.model)
        print(
            json.dumps(
                {
                    "name": name,
                    "gender": gender,
                    "method": None if skipped else ("tier2_homepage" if used_tier2 else "tier1_name"),
                    "skipped": skipped,
                },
                indent=2,
            )
        )
        return 0

    return process_people_batch(store, args.model, args.first_n, args.all)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
