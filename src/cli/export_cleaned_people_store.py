#!/usr/bin/env python3
"""Export a people-store copy with tags/flags removed."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import json
from typing import Any, Sequence

from util.data_store import DEFAULT_PEOPLE_REPO_PATH, DataStore


DEFAULT_OUTPUT_PATH = DEFAULT_PEOPLE_REPO_PATH / "people.json"
REMOVED_FIELDS = {"flags", "tags"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a cleaned people.json copy without tags or flags."
    )
    parser.add_argument(
        "--people-file",
        type=Path,
        default=None,
        help="Optional people JSONL path override. By default, DataStore chooses the store location.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def cleaned_person(person: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in person.items() if key not in REMOVED_FIELDS}


def export_cleaned_people(store: DataStore, output_path: Path) -> int:
    people = [cleaned_person(person) for person in store.list_people()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(people, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    return len(people)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = DataStore(path=args.people_file) if args.people_file is not None else DataStore()

    count = export_cleaned_people(store, args.output)
    print(f"Exported {count} cleaned people to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
