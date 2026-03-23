"""Infer country from affiliation for people records.

For people whose affiliation is set but country is missing, this script asks
an LLM to infer the country based only on the affiliation name (no web search).
Results are returned as ISO 3166-1 alpha-3 country codes.

All reads/writes of people records go through ``PeopleStore``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel

from llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response
from people import PeopleStore


INFER_COUNTRY_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_country_from_affiliation_batch.txt"
)
COUNTRY_SAVE_INTERVAL = 10
COUNTRY_CODE_RE = re.compile(r"^[A-Z]{3}$")


class CountryInferenceResult(BaseModel):
    """Result for one person's country inference."""
    original_name: str
    country: str | None = None


class CountryInferenceBatchResult(BaseModel):
    """Batch result with list of country assignments."""
    people: list[CountryInferenceResult]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer country from affiliation for people missing country field. "
            "Uses LLM (no web search). Countries returned as ISO 3166-1 alpha-3 codes."
        )
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional person name to process (case-insensitive match).",
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
        "--limit",
        type=int,
        default=None,
        help="Maximum number of people with missing country to process.",
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
        "--dry-run",
        action="store_true",
        help="Compute updates but do not write them.",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.first_n <= 0:
        parser.error("--first-n must be a positive integer")

    return args


def _load_prompt_template() -> str:
    try:
        return INFER_COUNTRY_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {INFER_COUNTRY_PROMPT_PATH}") from exc


def _is_valid_country_code(code: str | None) -> bool:
    """Check if code is a valid ISO 3166-1 alpha-3 country code."""
    if code is None:
        return True
    if not isinstance(code, str):
        return False
    return bool(COUNTRY_CODE_RE.match(code))


def _needs_country_inference(person: dict[str, Any]) -> bool:
    """Check if a person has affiliation but missing country."""
    affiliation = person.get("affiliation")
    country = person.get("country")
    
    # Must have non-empty affiliation
    if not isinstance(affiliation, str) or not affiliation.strip():
        return False
    
    # Must not have a country set
    if country is not None:
        return False
    
    return True


def _get_person_by_name(
    people: dict[str, dict[str, Any]], requested_name: str
) -> tuple[str, dict[str, Any]]:
    """Find person by name (case-insensitive)."""
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


def _batch_prompt(people_batch: list[dict[str, Any]]) -> str:
    """Create a batch prompt for country inference."""
    template = _load_prompt_template()
    
    # Build JSON array of people for the prompt
    people_for_prompt = []
    for person in people_batch:
        name = person.get("name", "")
        affiliation = person.get("affiliation", "")
        people_for_prompt.append({
            "name": name,
            "affiliation": affiliation,
            "country": None
        })
    
    people_json = json.dumps(people_for_prompt, indent=2, ensure_ascii=False)
    return template.format(people=people_json)


def infer_country_batch(people_batch: list[dict[str, Any]], model: str) -> dict[str, str | None]:
    """Query country for a batch of people (max 50) in a single LLM request.
    
    Args:
        people_batch: List of people records with affiliation but no country
        model: OpenAI model to use
        
    Returns:
        dict mapping person name to country code (ISO 3166-1 alpha-3) or None
    """
    if not people_batch:
        return {}
    
    if len(people_batch) > 50:
        raise ValueError(f"Batch size exceeds 50: {len(people_batch)}")
    
    batch_names = [p.get("name", "<unknown>") for p in people_batch]
    print(f"[LLM][Country Batch] Querying country for {len(batch_names)} people")
    
    try:
        parsed = parse_structured_response(
            input_text=_batch_prompt(people_batch),
            response_model=CountryInferenceBatchResult,
            description="infer country from affiliation batch",
            model=model,
        )
    except Exception as exc:
        print(f"[LLM][Country Batch] Error: {exc}")
        return {}
    
    result = {}
    for inference in parsed.people:
        name = inference.original_name
        country = inference.country
        
        # Validate country code if present
        if not _is_valid_country_code(country):
            print(f"[LLM][Country Batch][Warn] Invalid country code for {name}: {country}, treating as None")
            country = None
        
        result[name] = country
        if country:
            print(f"[LLM][Country Batch] Inferred country for {name}: {country}")
        else:
            print(f"[LLM][Country Batch] Could not infer country for {name}")
    
    return result


def _flush_country_updates(store: PeopleStore, pending_updates: list[dict[str, Any]]) -> int:
    """Write pending country updates to store."""
    if not pending_updates:
        return 0

    _, updated_count = store.update_many(pending_updates)
    print(f"[Store] Checkpointed {updated_count} country updates")
    pending_updates.clear()
    return updated_count


def process_people_batch(
    store: PeopleStore,
    model: str,
    first_n: int = 10,
    process_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Process a batch of people to infer missing countries.
    
    Args:
        store: PeopleStore instance
        model: OpenAI model to use
        first_n: Number of people needing country to process if not using --all
        process_all: If True, process all people needing country (ignores first_n)
        limit: Maximum total people to process
        dry_run: If True, compute updates but don't write them
    
    Returns:
        Exit code (0 for success)
    """
    people = store.list_people()
    
    # First filter to only people who need country inference
    people_needing_country: list[dict[str, Any]] = []
    for person in people:
        if _needs_country_inference(person):
            people_needing_country.append(person)
    
    print(f"[Batch] Found {len(people_needing_country)} people with affiliation but no country")
    
    # Then select which ones to process
    selected = people_needing_country if process_all else people_needing_country[:first_n]
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        selected = selected[:limit]
    
    print(f"[Batch] Processing {len(selected)} people")

    processed_count = 0
    skipped_count = 0
    pending_country_updates: list[dict[str, Any]] = []
    
    people_to_process = selected
    
    # Process in batches of up to 50
    batch_size = 50
    for batch_start in range(0, len(people_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(people_to_process))
        batch = people_to_process[batch_start:batch_end]
        
        print(f"[Batch] Processing records {batch_start + 1} to {batch_end} (batch size: {len(batch)})")
        
        # Run country inference batch query
        batch_results = infer_country_batch(batch, model)
        
        # Collect updates
        for person in batch:
            person_name = str(person.get("name", "<unknown>"))
            country = batch_results.get(person_name)
            
            if country is not None:
                # Only update if we got a country code
                pending_country_updates.append({"name": person_name, "country": country})
                processed_count += 1
            else:
                # No country inferred, skip this person
                print(f"[Skip] Could not infer country for {person_name}")
                skipped_count += 1
        
        # Checkpoint after processing this batch
        if len(pending_country_updates) >= COUNTRY_SAVE_INTERVAL and not dry_run:
            _flush_country_updates(store, pending_country_updates)
    
    # Final checkpoint
    if not dry_run:
        _flush_country_updates(store, pending_country_updates)
    elif pending_country_updates:
        print(f"[Dry-run] Would update {len(pending_country_updates)} people")
        for update in pending_country_updates:
            print(f"  - {update['name']}: {update.get('country')}")
        pending_country_updates.clear()

    print(f"[Batch] Completed. Processed: {processed_count}, skipped: {skipped_count}")
    return 0


def infer_country_for_person(
    store: PeopleStore,
    requested_name: str,
    model: str,
    dry_run: bool = False,
) -> tuple[str, str | None, bool]:
    """Infer country for a single person.
    
    Returns:
        (name, country_code, was_updated)
    """
    people = store.load()
    _, person = _get_person_by_name(people, requested_name)
    
    name = str(person.get("name", "<unknown>"))
    
    if not _needs_country_inference(person):
        existing_country = person.get("country")
        print(f"[Skip] Country already set for {name}: {existing_country}")
        return name, existing_country, False
    
    print(f"[Single] Inferring country for {name}")
    batch_results = infer_country_batch([person], model)
    country = batch_results.get(name)
    
    if country is not None and not dry_run:
        updated = store.update_person(name, {"country": country})
        print(f"[Store] Saved country for {updated['name']}: {country}")
        return str(updated["name"]), country, True
    elif country is not None and dry_run:
        print(f"[Dry-run] Would update {name} with country: {country}")
        return name, country, True
    else:
        print(f"[Skip] Could not infer country for {name}")
        return name, None, False


def main(argv: Sequence[str] | None = None) -> int:
    parser = parse_args(argv)
    store = PeopleStore(path=parser.people_file) if parser.people_file is not None else PeopleStore()

    if parser.name:
        name, country, updated = infer_country_for_person(
            store, parser.name, parser.model, dry_run=parser.dry_run
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "country": country,
                    "updated": updated,
                },
                indent=2,
            )
        )
        return 0

    return process_people_batch(
        store,
        parser.model,
        first_n=parser.first_n,
        process_all=parser.all,
        limit=parser.limit,
        dry_run=parser.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
