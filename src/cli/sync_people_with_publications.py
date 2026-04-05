"""Sync people records with publication summaries from filtered DBLP.

This module updates existing people with publication_summary attributes and
adds new people who have a minimum number of publications in the target venues.
Uses query_dblp.py to query and summarize publications.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
from datetime import datetime
from typing import Any

from util.data_store import DataStore
from util.query_dblp import DblpQueryEngine, create_publication_summary


def _format_summary(person_name: str, summary: dict[str, Any]) -> str:
    by_venue = ", ".join(
        f"{venue}: {count}" for venue, count in sorted(summary["by_venue"].items())
    )
    year_range = f"{summary['year_range'][0]}-{summary['year_range'][1]}" if summary["year_range"] else "n/a"
    return f"{person_name}: {summary['total']} papers ({by_venue}), years {year_range}"


def sync_single_person_publications(
    person_name: str,
    *,
    base_commit: str | None = None,
    current_year: int | None = None,
    max_years_back: int = 5,
) -> tuple[bool, dict[str, Any] | None]:
    """Recompute and store publication_summary for one existing person.

    Returns:
        Tuple of (changed, summary).
        - changed=True when the stored publication summary was updated.
        - summary=None when no target-venue publications were found.
    """
    normalized_name = person_name.strip()
    if not normalized_name:
        raise ValueError("person_name must be non-empty")

    if current_year is None:
        current_year = datetime.now().year

    store = DataStore()
    people = store.load(commit=base_commit)
    if normalized_name not in people:
        raise ValueError(f"No person found with name: {normalized_name}")

    engine = DblpQueryEngine(preload_index=True)
    summary = create_publication_summary(
        normalized_name,
        current_year=current_year,
        max_years_back=max_years_back,
        engine=engine,
    )

    existing_summary = people[normalized_name].get("publication_summary")
    if summary == existing_summary:
        return False, summary

    if summary:
        store.update(
            normalized_name,
            {"publication_summary": summary},
            base_commit=base_commit,
        )

    return bool(summary), summary


def sync_people_with_publications(
    current_year: int | None = None,
    max_years_back: int = 5,
    min_publications: int = 5,
    dry_run: bool = False,
    limit: int | None = None,
    recompute_all: bool = False,
) -> tuple[int, int, int, int]:
    """Update existing people and add prolific authors from DBLP.

    Existing people without publication_summary are backfilled. When
    recompute_all is True, every existing person's publication summary is
    recomputed and updated only if the value changed. In addition, authors not
    yet in the people store are added when they have at least min_publications
    papers in target venues from the configured year window.

    Args:
        current_year: The year to use as reference (default: current year)
        max_years_back: How many years back to include (default: 5)
        min_publications: Minimum number of publications required for new people
        dry_run: If True, don't write changes
        limit: Max number of existing people without publication data to process
        recompute_all: If True, recompute publication summaries for all existing people

    Returns:
        Tuple of (updated_existing_count, added_new_count, skipped_existing_count, total_people_before)
    """
    if current_year is None:
        current_year = datetime.now().year

    store = DataStore()
    people = store.load()
    known_people = set(people)
    min_year = current_year - max_years_back
    engine = DblpQueryEngine(preload_index=True)

    updated_existing_count = 0
    added_new_count = 0
    skipped_count = 0
    processed_count = 0
    pending_entries: list[dict[str, Any]] = []

    for person_name in sorted(people.keys(), key=str.casefold):
        existing_summary = people[person_name].get("publication_summary")
        if not recompute_all and existing_summary is not None:
            skipped_count += 1
            continue

        if limit is not None and processed_count >= limit:
            break

        processed_count += 1
        summary = create_publication_summary(
            person_name,
            current_year=current_year,
            max_years_back=max_years_back,
            engine=engine,
        )

        if summary == existing_summary:
            skipped_count += 1
            continue

        if summary:
            if dry_run:
                action = "refresh" if recompute_all and existing_summary is not None else "update"
                print(f"  [{action}] {_format_summary(person_name, summary)}")
            else:
                pending_entries.append({"name": person_name, "publication_summary": summary})
            updated_existing_count += 1
        else:
            skipped_count += 1

    prolific_authors = engine.query_prolific_authors_in_target_venues(
        min_publications=min_publications,
        min_year=min_year,
        max_year=current_year,
    )

    for person_name in sorted(prolific_authors.keys(), key=str.casefold):
        if person_name in known_people:
            continue

        summary = create_publication_summary(
            person_name,
            current_year=current_year,
            max_years_back=max_years_back,
            engine=engine,
        )
        if summary is None or summary["total"] < min_publications:
            continue

        if dry_run:
            print(f"  [add] {_format_summary(person_name, summary)}")
        else:
            pending_entries.append({"name": person_name, "publication_summary": summary})

        known_people.add(person_name)
        added_new_count += 1

    if not dry_run and pending_entries:
        added_new_count, updated_existing_count = store.update_many(pending_entries)

    return updated_existing_count, added_new_count, skipped_count, len(people)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update existing people with publication summaries and add prolific authors from DBLP"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Reference year (default: current year)",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="Years back to include (default: 5)",
    )
    parser.add_argument(
        "--min-publications",
        type=int,
        default=5,
        help="Minimum number of publications required to add a new person (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes",
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute publication summaries for all existing people and update only changed ones",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max existing people to process "
            "(default: 10 for --dry-run unless --recompute-all is used, no limit otherwise)"
        ),
    )

    args = parser.parse_args()

    limit = args.limit if args.limit is not None else (None if args.recompute_all or not args.dry_run else 10)
    updated_existing, added_new, skipped, total = sync_people_with_publications(
        current_year=args.year,
        max_years_back=args.years_back,
        min_publications=args.min_publications,
        dry_run=args.dry_run,
        limit=limit,
        recompute_all=args.recompute_all,
    )

    skipped_message = (
        f"{skipped} existing people were unchanged or still had no target-venue publications"
        if args.recompute_all
        else f"{skipped} existing people already had publication data"
    )
    print(
        "Updated "
        f"{updated_existing} existing people and added {added_new} new people "
        f"with >= {args.min_publications} target-venue publications "
        f"({skipped_message}; started with {total} people)"
    )
