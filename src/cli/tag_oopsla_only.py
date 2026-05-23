"""Tag people whose publication profile is exclusively OOPSLA.

This script scans existing people records and adds a configurable tag
(default: ``#oopslaonly``) to people whose ``publication_summary`` contains
only OOPSLA publications.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
from typing import Any, Sequence

from util.data_store import DataStore


def _normalized_tag(tag: str) -> str:
    normalized = tag.strip().casefold()
    if not normalized:
        raise ValueError("tag must be non-empty")
    if not normalized.startswith("#"):
        normalized = f"#{normalized}"
    if normalized == "#":
        raise ValueError("tag must contain at least one non-# character")
    return normalized


def _is_oopsla_only(person: dict[str, Any]) -> bool:
    summary = person.get("publication_summary")
    if not isinstance(summary, dict):
        return False

    total = summary.get("total")
    by_venue = summary.get("by_venue")
    if not isinstance(total, int) or total <= 0:
        return False
    if not isinstance(by_venue, dict) or not by_venue:
        return False

    oopsla_count = 0
    non_oopsla_positive = 0
    total_from_venues = 0

    for raw_venue, raw_count in by_venue.items():
        if not isinstance(raw_count, int) or raw_count < 0:
            return False

        venue = str(raw_venue).strip().casefold()
        total_from_venues += raw_count

        if venue == "oopsla":
            oopsla_count += raw_count
        elif raw_count > 0:
            non_oopsla_positive += 1

    if non_oopsla_positive > 0:
        return False
    if oopsla_count <= 0:
        return False

    # Ensure "all publications" in profile are OOPSLA according to summary.
    return oopsla_count == total and total_from_venues == total


def tag_oopsla_only_people(
    *,
    tag: str = "#oopslaonly",
    dry_run: bool = False,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Add *tag* to people whose publication summary is OOPSLA-only.

    Returns:
        Tuple of (tagged_count, already_tagged_count, considered_count)
    """
    normalized_tag = _normalized_tag(tag)

    store = DataStore()
    people = store.load()
    updates: list[dict[str, Any]] = []

    tagged_count = 0
    already_tagged_count = 0
    considered_count = 0

    for name in sorted(people.keys(), key=str.casefold):
        if limit is not None and considered_count >= limit:
            break

        person = people[name]
        if not _is_oopsla_only(person):
            continue

        considered_count += 1

        flags = person.get("flags")
        existing_flags: list[str]
        if isinstance(flags, list):
            existing_flags = [str(flag) for flag in flags]
        elif isinstance(flags, str):
            existing_flags = [flags]
        else:
            existing_flags = []

        normalized_existing = {
            flag.strip().casefold() if flag.strip().startswith("#") else f"#{flag.strip().casefold()}"
            for flag in existing_flags
            if flag.strip()
        }

        if normalized_tag in normalized_existing:
            already_tagged_count += 1
            continue

        if dry_run:
            print(f"[tag] {name} -> {normalized_tag}")
        else:
            updates.append({"name": name, "flags": [normalized_tag]})
        tagged_count += 1

    if not dry_run and updates:
        store.update_many(updates)

    return tagged_count, already_tagged_count, considered_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search for people with publication profiles exclusively at OOPSLA "
            "and add a tag to them"
        )
    )
    parser.add_argument(
        "--tag",
        default="#oopslaonly",
        help="Tag to add to matching people (default: #oopslaonly)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matches without writing changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of matching OOPSLA-only people to process",
    )

    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    return args


if __name__ == "__main__":
    args = parse_args()
    tagged, already_tagged, considered = tag_oopsla_only_people(
        tag=args.tag,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    action = "Would tag" if args.dry_run else "Tagged"
    print(
        f"{action} {tagged} people with {args.tag}; "
        f"{already_tagged} already had the tag "
        f"(considered {considered} OOPSLA-only profiles)"
    )
