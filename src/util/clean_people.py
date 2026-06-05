"""Shared person-cleaning helpers used by web and script workflows.

This module contains the LLM-backed affiliation/country cleanup logic that was
previously embedded in the clean_people CLI tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from util.data_store import DataStore
from util.llm_queries import parse_structured_response


COUNTRY_CODE_RE = re.compile(r"^[A-Z]{3}$")
CLEAN_AFFILIATION_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "clean_affiliation.txt"


class CleanAffiliationResult(BaseModel):
    affiliation: str
    country: str | None = None


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


def clean_single(
    store: DataStore,
    name: str,
    model: str,
    base_commit: str | None = None,
) -> tuple[bool, str]:
    people = store.load(commit=base_commit)
    original_name, entry = _get_person_by_name(people, name)
    cleaned = clean_record(entry, model)

    if cleaned == entry:
        return False, original_name

    store.replace_person(original_name, cleaned, base_commit=base_commit)
    return True, cleaned["name"]
