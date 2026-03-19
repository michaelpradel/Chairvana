"""Utilities for reading and updating the people JSONL store.

This module is the single point of entry for writes to `data/people.jsonl`.
Each person is uniquely identified by the `name` field.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PEOPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "people.jsonl"
DEFAULT_PEOPLE_REPO_DIRNAME = ".people_repo"


def _merge_values(existing: Any, new_value: Any) -> Any:
    if isinstance(existing, dict) and isinstance(new_value, dict):
        return _merge_dict(existing, new_value)

    if isinstance(existing, list) and isinstance(new_value, list):
        merged = list(existing)
        seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged}
        for item in new_value:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged

    return new_value


def _merge_dict(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if key in merged:
            merged[key] = _merge_values(merged[key], value)
        else:
            merged[key] = value
    return merged


def _extract_latest_pc_year(record: dict[str, Any]) -> int | None:
    memberships = record.get("pc_memberships")
    if not isinstance(memberships, list):
        return None

    years: list[int] = []
    for membership in memberships:
        if not isinstance(membership, dict):
            continue
        year = membership.get("year")
        if isinstance(year, int):
            years.append(year)

    if not years:
        return None
    return max(years)


def _merge_person_record(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = _merge_dict(existing, updates)

    existing_aff = existing.get("affiliation")
    update_aff = updates.get("affiliation")
    existing_year = _extract_latest_pc_year(existing)
    update_year = _extract_latest_pc_year(updates)

    # Keep affiliation single-valued and prefer data from newer conferences.
    if isinstance(existing_aff, str) and existing_aff.strip():
        if not isinstance(update_aff, str) or not update_aff.strip():
            merged["affiliation"] = existing_aff
        elif existing_year is not None and update_year is not None and update_year < existing_year:
            merged["affiliation"] = existing_aff

    merged.pop("affiliations", None)
    return merged


class PeopleStore:
    """Read and update person records stored as JSONL."""

    def __init__(self, path: Path | None = None, repo_dir: Path | None = None) -> None:
        self.path = path or DEFAULT_PEOPLE_PATH
        self.repo_dir = repo_dir or (self.path.parent / DEFAULT_PEOPLE_REPO_DIRNAME)

    def load(self) -> dict[str, dict[str, Any]]:
        people: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return people

        with self.path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if not isinstance(record, dict):
                    continue

                name = record.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue

                normalized_name = name.strip()
                existing = people.get(normalized_name)
                if existing is None:
                    people[normalized_name] = {"name": normalized_name, **record}
                else:
                    people[normalized_name] = _merge_person_record(
                        existing,
                        {"name": normalized_name, **record},
                    )

        return people

    def list_people(self) -> list[dict[str, Any]]:
        people = self.load()
        return [people[name] for name in sorted(people, key=str.casefold)]

    def update_many(self, entries: Iterable[dict[str, Any]]) -> tuple[int, int]:
        people = self.load()
        added = 0
        updated = 0

        for entry in entries:
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Each entry must contain a non-empty 'name': {entry}")

            normalized_name = name.strip()
            normalized_entry = {"name": normalized_name, **entry}

            if normalized_name in people:
                people[normalized_name] = _merge_person_record(
                    people[normalized_name],
                    normalized_entry,
                )
                updated += 1
            else:
                people[normalized_name] = normalized_entry
                added += 1

        self._write_all(people)
        return added, updated

    def update(self, name: str, updates: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must be non-empty")

        people = self.load()
        merged_entry = {"name": normalized_name, **updates}

        is_new = normalized_name not in people
        if is_new:
            people[normalized_name] = merged_entry
        else:
            people[normalized_name] = _merge_person_record(
                people[normalized_name],
                merged_entry,
            )

        self._write_all(people)
        return is_new, people[normalized_name]

    def update_person(self, original_name: str, updates: dict[str, Any]) -> dict[str, Any]:
        normalized_original_name = original_name.strip()
        if not normalized_original_name:
            raise ValueError("original_name must be non-empty")

        people = self.load()
        existing = people.get(normalized_original_name)
        if existing is None:
            raise ValueError(f"No person found with name: {normalized_original_name}")

        updated = dict(existing)
        updated.update(updates)

        new_name = updated.get("name")
        if not isinstance(new_name, str) or not new_name.strip():
            raise ValueError("Updated record must contain a non-empty 'name'")
        normalized_new_name = new_name.strip()
        updated["name"] = normalized_new_name

        affiliation = updated.get("affiliation")
        if isinstance(affiliation, str):
            normalized_affiliation = affiliation.strip()
            updated["affiliation"] = normalized_affiliation
        updated.pop("affiliations", None)

        if normalized_new_name != normalized_original_name and normalized_new_name in people:
            raise ValueError(f"A person with name '{normalized_new_name}' already exists")

        del people[normalized_original_name]
        people[normalized_new_name] = updated

        self._write_all(people)
        return updated

    def _write_all(self, people: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sorted_names = sorted(people, key=str.casefold)

        with self.path.open("w", encoding="utf-8") as file_obj:
            for name in sorted_names:
                record = people[name]
                record.pop("affiliations", None)
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._commit_people_file(len(sorted_names))

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            "git",
            f"--git-dir={self.repo_dir / '.git'}",
            f"--work-tree={self.path.parent}",
            *args,
        ]
        return subprocess.run(command, capture_output=True, text=True, check=check)

    def _ensure_local_repo(self) -> None:
        git_dir = self.repo_dir / ".git"
        if git_dir.exists():
            return

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(self.repo_dir)], check=True, capture_output=True, text=True)
        self._git("config", "user.name", "People Store Bot")
        self._git("config", "user.email", "people-store@local")

    def _commit_people_file(self, people_count: int) -> None:
        self._ensure_local_repo()

        people_filename = self.path.name
        self._git("add", "--", people_filename)

        diff_status = self._git("diff", "--cached", "--quiet", "--", people_filename, check=False)
        if diff_status.returncode == 0:
            return
        if diff_status.returncode != 1:
            raise RuntimeError(diff_status.stderr.strip() or "git diff failed")

        self._git("commit", "-m", f"Update people.jsonl ({people_count} people)")
