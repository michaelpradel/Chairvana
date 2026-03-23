"""Auto-complete missing affiliation/country data in the people store.

Two-stage process:
1. For people with affiliation but no country: infer country via LLM (no web search).
2. For people missing affiliation: find homepage + infer affiliation/country via web search.

For stage 2, to reduce name ambiguity, the prompt includes a few recent paper titles from DBLP.
All reads/writes of people records go through ``PeopleStore``.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

from llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response
from people import PeopleStore
from query_dblp import DblpQueryEngine, get_target_publications_for_author


AUTO_COMPLETE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "auto_complete_affiliation.txt"
INFER_COUNTRY_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "infer_country_from_affiliation_batch.txt"
)
COUNTRY_CODE_RE = re.compile(r"^[A-Z]{3}$")
URL_ADAPTER = TypeAdapter(HttpUrl)


class AutoCompleteResult(BaseModel):
    homepage: str
    affiliation: str
    country: str | None = None


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
            "Auto-complete missing affiliation/country fields via LLM web search, "
            "using recent DBLP paper titles as disambiguation context."
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
        help="Maximum number of people with missing affiliation to process.",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="Years of DBLP papers to include as context (default: 5).",
    )
    parser.add_argument(
        "--max-paper-titles",
        type=int,
        default=5,
        help="Maximum number of recent paper titles to include in prompt context (default: 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates but do not write them.",
    )
    parser.add_argument(
        "--skip-country-inference",
        action="store_true",
        help="Skip the country inference stage (only complete missing affiliations).",
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.years_back <= 0:
        parser.error("--years-back must be a positive integer")
    if args.max_paper_titles <= 0:
        parser.error("--max-paper-titles must be a positive integer")

    return args


def _load_country_prompt_template() -> str:
    """Load the country inference prompt template."""
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


def _batch_country_prompt(people_batch: list[dict[str, Any]]) -> str:
    """Create a batch prompt for country inference."""
    template = _load_country_prompt_template()
    
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


def _infer_country_batch(people_batch: list[dict[str, Any]], model: str) -> dict[str, str | None]:
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
    print(f"[Country] Inferring country for {len(batch_names)} people")
    
    try:
        parsed = parse_structured_response(
            input_text=_batch_country_prompt(people_batch),
            response_model=CountryInferenceBatchResult,
            description="infer country from affiliation batch",
            model=model,
        )
    except Exception as exc:
        print(f"[Country][Warn] LLM request failed: {exc}")
        return {}
    
    result = {}
    for inference in parsed.people:
        name = inference.original_name
        country = inference.country
        
        # Validate country code if present
        if not _is_valid_country_code(country):
            print(f"[Country][Warn] Invalid country code for {name}: {country}, skipping")
            country = None
        
        result[name] = country
    
    return result


def _infer_and_save_countries(
    store: PeopleStore,
    model: str,
    dry_run: bool,
) -> int:
    """Infer countries for people with affiliation but no country.
    
    Returns:
        Number of people updated
    """
    people = store.load()
    
    # Filter to people needing country inference
    people_needing_country: list[dict[str, Any]] = []
    for person in sorted(people.values(), key=lambda p: str(p.get("name", "")).casefold()):
        if _needs_country_inference(person):
            people_needing_country.append(person)
    
    if not people_needing_country:
        print("[Country] No people found with affiliation but missing country")
        return 0
    
    print(f"[Country] Found {len(people_needing_country)} people with affiliation but no country")
    
    total_updated = 0
    pending_updates: list[dict[str, Any]] = []
    
    # Process in batches of up to 50
    batch_size = 50
    for batch_start in range(0, len(people_needing_country), batch_size):
        batch_end = min(batch_start + batch_size, len(people_needing_country))
        batch = people_needing_country[batch_start:batch_end]
        
        print(f"[Country] Processing batch {batch_start // batch_size + 1} ({len(batch)} people)")
        
        # Run country inference
        batch_results = _infer_country_batch(batch, model)
        
        # Collect updates
        for person in batch:
            person_name = str(person.get("name", "<unknown>"))
            country = batch_results.get(person_name)
            
            if country is not None:
                pending_updates.append({"name": person_name, "country": country})
                total_updated += 1
    
    # Save all updates
    if pending_updates:
        if dry_run:
            print(f"[Dry-run] Would update {len(pending_updates)} people with country info")
            for update in pending_updates:
                print(f"  {update['name']}: {update['country']}")
        else:
            _, updated_count = store.update_many(pending_updates)
            print(f"[Country] Saved {updated_count} country updates")
    
    return total_updated



def _load_prompt_template() -> str:
    try:
        return AUTO_COMPLETE_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {AUTO_COMPLETE_PROMPT_PATH}") from exc


def _normalize_optional_homepage(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""

    try:
        return str(URL_ADAPTER.validate_python(candidate))
    except ValidationError:
        print(f"[Warn] Invalid homepage returned by LLM: {value!r}")
        return ""


def _normalize_country_code(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().upper()
    if not normalized:
        return None
    if not COUNTRY_CODE_RE.fullmatch(normalized):
        print(f"[Warn] Invalid country code returned by LLM: {value!r}")
        return None
    return normalized


def _has_missing_affiliation(person: dict[str, Any]) -> bool:
    affiliation = person.get("affiliation")
    return not isinstance(affiliation, str) or not affiliation.strip()


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


def _recent_paper_titles(
    person_name: str,
    engine: DblpQueryEngine,
    *,
    current_year: int,
    years_back: int,
    max_titles: int,
) -> list[str]:
    publications = get_target_publications_for_author(
        person_name,
        engine,
        current_year=current_year,
        max_years_back=years_back,
    )

    publications = sorted(
        publications,
        key=lambda pub: (pub.year if pub.year is not None else -1, pub.title or ""),
        reverse=True,
    )

    titles: list[str] = []
    seen: set[str] = set()
    for publication in publications:
        if not isinstance(publication.title, str):
            continue
        title = publication.title.strip()
        if not title:
            continue
        if title in seen:
            continue
        seen.add(title)
        titles.append(title)
        if len(titles) >= max_titles:
            break
    return titles


def _build_prompt(
    person: dict[str, Any],
    prompt_template: str,
    paper_titles: list[str],
) -> str:
    name = str(person.get("name", "")).strip()
    homepage_value = person.get("homepage")
    existing_homepage = homepage_value.strip() if isinstance(homepage_value, str) else ""
    papers_block = "\n".join(f"- {title}" for title in paper_titles) if paper_titles else "- (none found)"
    return prompt_template.format(
        name=name,
        existing_homepage=existing_homepage,
        recent_papers=papers_block,
    )


def _infer_affiliation_and_country(
    person: dict[str, Any],
    *,
    model: str,
    prompt_template: str,
    engine: DblpQueryEngine,
    current_year: int,
    years_back: int,
    max_paper_titles: int,
) -> dict[str, Any] | None:
    name = str(person.get("name", "")).strip()
    if not name:
        return None

    paper_titles = _recent_paper_titles(
        name,
        engine,
        current_year=current_year,
        years_back=years_back,
        max_titles=max_paper_titles,
    )
    prompt = _build_prompt(person, prompt_template, paper_titles)

    try:
        parsed = parse_structured_response(
            input_text=prompt,
            response_model=AutoCompleteResult,
            description="auto-complete person info",
            model=model,
            tools=[{"type": "web_search"}],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Warn] LLM request failed for {name}: {exc}")
        return None

    affiliation = parsed.affiliation.strip()
    if not affiliation:
        print(f"[Skip] Empty affiliation returned for {name}")
        return None

    updates: dict[str, Any] = {
        "name": name,
        "affiliation": affiliation,
    }

    homepage = _normalize_optional_homepage(parsed.homepage)
    if homepage:
        updates["homepage"] = homepage

    country = _normalize_country_code(parsed.country)
    if country is not None:
        updates["country"] = country

    return updates


def _iter_targets_for_affiliation(
    people: dict[str, dict[str, Any]],
    *,
    requested_name: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if requested_name is not None:
        _, person = _get_person_by_name(people, requested_name)
        if not _has_missing_affiliation(person):
            print(f"[Skip] Affiliation already set for {person.get('name', '<unknown>')}")
            return []
        return [person]

    targets = [
        people[name]
        for name in sorted(people, key=str.casefold)
        if _has_missing_affiliation(people[name])
    ]
    return targets[:limit] if limit is not None else targets


def auto_complete_affiliations(
    *,
    store: PeopleStore,
    model: str,
    requested_name: str | None,
    limit: int | None,
    years_back: int,
    max_paper_titles: int,
    skip_country_inference: bool,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    """Process country inference and affiliation completion.
    
    Returns:
        (country_updated, affiliation_processed, affiliation_changed, affiliation_skipped)
    """
    # Stage 1: Infer countries for people with affiliation but no country
    country_updated = 0
    if not skip_country_inference:
        country_updated = _infer_and_save_countries(store, model, dry_run)
    
    # Reload people after country inference stage
    people = store.load()
    targets = _iter_targets_for_affiliation(people, requested_name=requested_name, limit=limit)

    if not targets:
        return country_updated, 0, 0, 0

    prompt_template = _load_prompt_template()
    engine = DblpQueryEngine(preload_index=True)
    current_year = datetime.now().year

    pending_updates: list[dict[str, Any]] = []
    skipped = 0

    for index, person in enumerate(targets, start=1):
        name = str(person.get("name", "<unknown>"))
        print(f"[LLM] ({index}/{len(targets)}) Completing affiliation for: {name}")

        updates = _infer_affiliation_and_country(
            person,
            model=model,
            prompt_template=prompt_template,
            engine=engine,
            current_year=current_year,
            years_back=years_back,
            max_paper_titles=max_paper_titles,
        )
        if updates is None:
            skipped += 1
            continue

        pending_updates.append(updates)
        print(
            "[Plan] "
            f"{name} -> affiliation={updates.get('affiliation')!r}, "
            f"country={updates.get('country', '<none>')!r}, "
            f"homepage={updates.get('homepage', '<none>')!r}"
        )

    if dry_run:
        for update in pending_updates:
            print(json.dumps(update, ensure_ascii=False))
        return country_updated, len(targets), len(pending_updates), skipped

    if not pending_updates:
        return country_updated, len(targets), 0, skipped

    added, updated = store.update_many(pending_updates)
    return country_updated, len(targets), added + updated, skipped


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = PeopleStore(path=args.people_file) if args.people_file is not None else PeopleStore()

    country_updated, processed, changed, skipped = auto_complete_affiliations(
        store=store,
        model=args.model,
        requested_name=args.name,
        limit=args.limit,
        years_back=args.years_back,
        max_paper_titles=args.max_paper_titles,
        skip_country_inference=args.skip_country_inference,
        dry_run=args.dry_run,
    )

    print(f"Country inference: {country_updated} updated")
    print(
        f"Affiliation completion: processed {processed}; "
        f"changed {changed}; skipped {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())