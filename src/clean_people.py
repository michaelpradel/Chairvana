"""Clean person affiliations and country codes in people.jsonl.

This script asks an LLM to normalize affiliation data for either:
- one entry identified by name, or
- all entries in the people store.

All writes go through ``PeopleStore`` so updates are committed to the
internal local git repository used by Chairvana.
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


COUNTRY_CODE_RE = re.compile(r"^[A-Z]{3}$")
CLEAN_AFFILIATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "clean_affiliation.txt"


class CleanAffiliationResult(BaseModel):
    affiliation: str
    country: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean affiliation/country fields in people.jsonl via an LLM."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Clean one entry by exact person name.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean all entries in the people file.",
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
    if bool(args.name) == bool(args.all):
        parser.error("Specify exactly one of --name or --all")

    return args


def _cleaning_prompt(person: dict[str, Any]) -> str:
    try:
        template = CLEAN_AFFILIATION_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {CLEAN_AFFILIATION_PROMPT_PATH}") from exc
    return template.format(person=json.dumps(person, ensure_ascii=False))


def _normalize_country_code(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().upper()
    if not normalized:
        return None
    if not COUNTRY_CODE_RE.fullmatch(normalized):
        raise ValueError(f"Country code is not ISO-3166 alpha-3: {value!r}")
    return normalized


def clean_record(person: dict[str, Any], model: str) -> dict[str, Any]:
    affiliation = person.get("affiliation")
    if not isinstance(affiliation, str) or not affiliation.strip():
        return dict(person)

    parsed = parse_structured_response(
        input_text=_cleaning_prompt(person),
        response_model=CleanAffiliationResult,
        description="clean affiliation",
        model=model,
    )

    cleaned_affiliation = parsed.affiliation.strip()
    if not cleaned_affiliation:
        raise ValueError(f"Model returned an empty affiliation for person: {person.get('name')}")

    cleaned = dict(person)
    cleaned["affiliation"] = cleaned_affiliation

    country = _normalize_country_code(parsed.country)
    if country is not None:
        cleaned["country"] = country
    else:
        existing_country = cleaned.get("country")
        if isinstance(existing_country, str):
            normalized_existing = _normalize_country_code(existing_country)
            if normalized_existing is not None:
                cleaned["country"] = normalized_existing
            else:
                cleaned.pop("country", None)
        else:
            cleaned.pop("country", None)

    return cleaned


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


def clean_single(store: PeopleStore, name: str, model: str, base_commit: str | None = None) -> tuple[bool, str]:
    people = store.load(commit=base_commit)
    original_name, entry = _get_person_by_name(people, name)
    cleaned = clean_record(entry, model)

    if cleaned == entry:
        return False, original_name

    store.replace_person(original_name, cleaned, base_commit=base_commit)
    return True, cleaned["name"]


def clean_all(store: PeopleStore, model: str) -> tuple[int, int]:
    people = store.load()
    cleaned_records: list[dict[str, Any]] = []
    changed_count = 0

    for name in sorted(people, key=str.casefold):
        original = people[name]
        cleaned = clean_record(original, model)
        if cleaned != original:
            changed_count += 1
        cleaned_records.append(cleaned)

    if not cleaned_records:
        return 0, 0

    replaced, _ = store.replace_many(cleaned_records)
    return replaced, changed_count


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = PeopleStore(path=args.people_file) if args.people_file is not None else PeopleStore()

    if args.name:
        changed, cleaned_name = clean_single(store, args.name, args.model)
        if changed:
            print(f"Cleaned and saved: {cleaned_name}")
        else:
            print(f"No changes required: {cleaned_name}")
        return 0

    replaced, changed = clean_all(store, args.model)
    print(f"Processed {replaced} entries; changed {changed} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
