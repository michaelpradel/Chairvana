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
from people import PeopleStore
from web_search import find_homepage_and_email


TIER1_NAME_GENDER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_gender_tier1_name.txt"
)
TIER2_HOMEPAGE_GENDER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_gender_tier2_homepage.txt"
)


class Tier1NameGenderResult(BaseModel):
    classification: Literal["very_likely_male", "very_likely_female", "unclear"]


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
        help="Optional people JSONL path override. By default, people.py chooses the store location.",
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
        model=model,
    )
    print(f"[LLM][Tier 1] Classification for {name}: {parsed.classification}")

    if parsed.classification == "very_likely_male":
        return "male"
    if parsed.classification == "very_likely_female":
        return "female"
    return None


def infer_gender_from_homepage(name: str, homepage: str, model: str) -> str:
    print(f"[LLM][Tier 2] Querying homepage-based gender for: {name} ({homepage})")
    parsed = parse_structured_response(
        input_text=_tier2_prompt(name, homepage),
        response_model=Tier2HomepageGenderResult,
        model=model,
        tools=[{"type": "web_search"}],
    )
    print(f"[LLM][Tier 2] Gender for {name}: {parsed.gender}")
    return parsed.gender


def _ensure_homepage(person: dict[str, Any], store: PeopleStore) -> dict[str, Any]:
    homepage = person.get("homepage")
    if isinstance(homepage, str) and homepage.strip():
        return person

    name = person.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Person record has no valid name")

    print(f"[Web] No homepage for {name}; querying homepage/email")
    resolved = find_homepage_and_email(name)
    updates = {
        "homepage": resolved.homepage,
        "email": resolved.email,
    }
    updated = store.update_person(name, updates)
    print(f"[Store] Saved homepage/email for {name}")
    return updated


def infer_and_store_gender_for_person(
    store: PeopleStore,
    person: dict[str, Any],
    model: str,
) -> tuple[str, str, bool]:
    name = person.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Person record must contain a non-empty 'name'")
    canonical_name = name.strip()

    tier1_gender = infer_gender_tier1(canonical_name, model)
    if tier1_gender is not None:
        updated = store.update_person(canonical_name, {"gender": tier1_gender})
        print(f"[Store] Saved gender for {canonical_name}: {tier1_gender}")
        return updated["name"], tier1_gender, False

    person_with_homepage = _ensure_homepage(person, store)
    homepage = person_with_homepage.get("homepage")
    if not isinstance(homepage, str) or not homepage.strip():
        raise ValueError(f"Could not obtain homepage for person: {canonical_name}")

    tier2_gender = infer_gender_from_homepage(canonical_name, homepage, model)
    updated = store.update_person(canonical_name, {"gender": tier2_gender})
    print(f"[Store] Saved gender for {canonical_name}: {tier2_gender}")
    return updated["name"], tier2_gender, True


def infer_and_store_gender(
    store: PeopleStore,
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


def process_people_batch(store: PeopleStore, model: str, first_n: int, process_all: bool) -> int:
    people = store.list_people()
    selected = people if process_all else people[:first_n]
    print(f"[Batch] Processing {len(selected)} people")

    processed_count = 0
    skipped_count = 0
    for person in selected:
        person_name = str(person.get("name", "<unknown>"))
        print(f"[Batch] Processing person: {person_name}")
        _, _, _, skipped = infer_and_store_gender(store, person_name, model)
        if skipped:
            skipped_count += 1
        else:
            processed_count += 1

    print(f"[Batch] Completed. Updated: {processed_count}, skipped: {skipped_count}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = PeopleStore(path=args.people_file) if args.people_file is not None else PeopleStore()

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
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc
