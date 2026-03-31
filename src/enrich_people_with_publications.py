"""Sync people records with publication summaries from filtered DBLP.

This module updates existing people with publication_summary attributes and
adds new people who have a minimum number of publications in the target venues.
Uses query_dblp.py to query and summarize publications.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from data_store import DataStore
from query_dblp import DblpQueryEngine, create_publication_summary


def _format_summary(person_name: str, summary: dict[str, Any]) -> str:
    by_venue = ", ".join(
        f"{venue}: {count}" for venue, count in sorted(summary["by_venue"].items())
    )
    year_range = f"{summary['year_range'][0]}-{summary['year_range'][1]}" if summary["year_range"] else "n/a"
    return f"{person_name}: {summary['total']} papers ({by_venue}), years {year_range}"


def sync_people_with_publications(
    current_year: int | None = None,
    max_years_back: int = 5,
    min_publications: int = 5,
    dry_run: bool = False,
    limit: int | None = None,
) -> tuple[int, int, int, int]:
    """Update existing people and add prolific authors from DBLP.

    Existing people without publication_summary are backfilled. In addition,
    authors not yet in the people store are added when they have at least
    min_publications papers in target venues from the configured year window.

    Args:
        current_year: The year to use as reference (default: current year)
        max_years_back: How many years back to include (default: 5)
        min_publications: Minimum number of publications required for new people
        dry_run: If True, don't write changes
        limit: Max number of existing people without publication data to process

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
        if "publication_summary" in people[person_name]:
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

        if summary:
            if dry_run:
                print(f"  [update] {_format_summary(person_name, summary)}")
            else:
                pending_entries.append({"name": person_name, "publication_summary": summary})
            updated_existing_count += 1

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
        "--limit",
        type=int,
        default=None,
        help="Max people without existing data to process (default: 10 for --dry-run, no limit otherwise)",
    )

    args = parser.parse_args()

    limit = args.limit if args.limit is not None else (10 if args.dry_run else None)
    updated_existing, added_new, skipped, total = sync_people_with_publications(
        current_year=args.year,
        max_years_back=args.years_back,
        min_publications=args.min_publications,
        dry_run=args.dry_run,
        limit=limit,
    )

    print(
        "Updated "
        f"{updated_existing} existing people and added {added_new} new people "
        f"with >= {args.min_publications} target-venue publications "
        f"({skipped} existing people already had publication data; started with {total} people)"
    )
