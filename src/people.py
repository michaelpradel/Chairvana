"""Utilities for reading and updating the people JSONL store.

This module is the single point of entry for writes to `data/people.jsonl`.
Each person is uniquely identified by the `name` field.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PEOPLE_REPO_DIRNAME = ".people_repo"
DEFAULT_PEOPLE_REPO_PATH = Path(__file__).resolve().parent.parent / "data" / DEFAULT_PEOPLE_REPO_DIRNAME
DEFAULT_PEOPLE_PATH = DEFAULT_PEOPLE_REPO_PATH / "people.jsonl"


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


def _normalize_flags(raw_flags: Any) -> list[str] | None:
    if raw_flags is None:
        return None

    tokens: list[str] = []
    if isinstance(raw_flags, str):
        tokens = [token.strip() for token in re.split(r"[,\s]+", raw_flags) if token.strip()]
    elif isinstance(raw_flags, list):
        tokens = [str(token).strip() for token in raw_flags if str(token).strip()]
    else:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if not lowered.startswith("#"):
            lowered = f"#{lowered}"

        if lowered == "#":
            continue

        if lowered not in seen:
            seen.add(lowered)
            normalized.append(lowered)

    return normalized or None


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

    normalized_flags = _normalize_flags(merged.get("flags"))
    if normalized_flags:
        merged["flags"] = normalized_flags
    else:
        merged.pop("flags", None)

    merged.pop("affiliations", None)
    return merged


class PeopleStore:
    """Read and update person records stored as JSONL."""

    def __init__(self, path: Path | None = None, repo_dir: Path | None = None) -> None:
        self.path = path or DEFAULT_PEOPLE_PATH
        if repo_dir is not None:
            self.repo_dir = repo_dir
        elif self.path.parent.name == DEFAULT_PEOPLE_REPO_DIRNAME:
            self.repo_dir = self.path.parent
        else:
            self.repo_dir = self.path.parent / DEFAULT_PEOPLE_REPO_DIRNAME

    def load(self, commit: str | None = None) -> dict[str, dict[str, Any]]:
        if commit is None:
            if not self.path.exists():
                return {}

            with self.path.open("r", encoding="utf-8") as file_obj:
                return self._load_from_lines(file_obj)

        normalized_commit = self.resolve_history_commit(commit)
        content = self._read_people_file_from_commit(normalized_commit)
        return self._load_from_lines(content.splitlines())

    def list_people(self, commit: str | None = None) -> list[dict[str, Any]]:
        people = self.load(commit=commit)
        return [people[name] for name in sorted(people, key=str.casefold)]

    def list_history(self) -> list[dict[str, str]]:
        self._ensure_local_repo()
        people_filename = self._people_repo_path()
        log_result = self._git(
            "log",
            "--format=%H%x1f%h%x1f%cs%x1f%s",
            "--",
            people_filename,
            check=False,
        )
        if log_result.returncode != 0:
            stderr = log_result.stderr.strip()
            if "does not have any commits yet" in stderr:
                return []
            raise RuntimeError(stderr or "git log failed")

        history: list[dict[str, str]] = []
        for line in log_result.stdout.splitlines():
            if not line.strip():
                continue

            commit_hash, short_hash, commit_date, subject = line.split("\x1f", maxsplit=3)
            history.append(
                {
                    "commit": commit_hash,
                    "short_commit": short_hash,
                    "date": commit_date,
                    "subject": subject,
                }
            )
        return history

    def get_history_state(self, commit: str | None = None) -> dict[str, Any]:
        history = self.list_history()
        if not history:
            return {
                "entries": history,
                "current_commit": None,
                "current_entry": None,
                "older_commit": None,
                "newer_commit": None,
                "is_head": True,
            }

        current_commit = history[0]["commit"] if commit is None else self.resolve_history_commit(commit)
        current_index = next((index for index, entry in enumerate(history) if entry["commit"] == current_commit), None)
        if current_index is None:
            raise ValueError(f"Commit {current_commit} is not in the history of {self._people_repo_path()}")

        older_commit = history[current_index + 1]["commit"] if current_index + 1 < len(history) else None
        newer_commit = history[current_index - 1]["commit"] if current_index > 0 else None
        return {
            "entries": history,
            "current_commit": current_commit,
            "current_entry": history[current_index],
            "older_commit": older_commit,
            "newer_commit": newer_commit,
            "is_head": current_index == 0,
        }

    def resolve_history_commit(self, commit: str) -> str:
        normalized_commit = commit.strip()
        if not normalized_commit:
            raise ValueError("history commit must be non-empty")

        history = self.list_history()
        matching_commits = [entry["commit"] for entry in history if entry["commit"].startswith(normalized_commit)]
        if len(matching_commits) == 1:
            return matching_commits[0]
        if len(matching_commits) > 1:
            raise ValueError(f"Ambiguous history commit: {commit}")
        raise ValueError(f"Unknown history commit: {commit}")

    def update_many(
        self,
        entries: Iterable[dict[str, Any]],
        *,
        base_commit: str | None = None,
    ) -> tuple[int, int]:
        self._prepare_write_base(base_commit)
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

    def update(
        self,
        name: str,
        updates: dict[str, Any],
        *,
        base_commit: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must be non-empty")

        self._prepare_write_base(base_commit)
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

    def update_person(
        self,
        original_name: str,
        updates: dict[str, Any],
        *,
        base_commit: str | None = None,
    ) -> dict[str, Any]:
        normalized_original_name = original_name.strip()
        if not normalized_original_name:
            raise ValueError("original_name must be non-empty")

        self._prepare_write_base(base_commit)
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

        normalized_flags = _normalize_flags(updated.get("flags"))
        if normalized_flags:
            updated["flags"] = normalized_flags
        else:
            updated.pop("flags", None)

        updated.pop("affiliations", None)

        if normalized_new_name != normalized_original_name and normalized_new_name in people:
            raise ValueError(f"A person with name '{normalized_new_name}' already exists")

        del people[normalized_original_name]
        people[normalized_new_name] = updated

        self._write_all(people)
        return updated

    def replace_person(
        self,
        original_name: str,
        replacement: dict[str, Any],
        *,
        base_commit: str | None = None,
    ) -> dict[str, Any]:
        normalized_original_name = original_name.strip()
        if not normalized_original_name:
            raise ValueError("original_name must be non-empty")

        self._prepare_write_base(base_commit)
        people = self.load()
        if normalized_original_name not in people:
            raise ValueError(f"No person found with name: {normalized_original_name}")

        normalized = self._normalize_person_record(replacement)
        new_name = normalized["name"]
        if new_name != normalized_original_name and new_name in people:
            raise ValueError(f"A person with name '{new_name}' already exists")

        del people[normalized_original_name]
        people[new_name] = normalized

        self._write_all(people)
        return normalized

    def replace_many(
        self,
        replacements: Iterable[dict[str, Any]],
        *,
        base_commit: str | None = None,
    ) -> tuple[int, int]:
        self._prepare_write_base(base_commit)
        people = self.load()
        replaced = 0
        renamed = 0

        for replacement in replacements:
            normalized = self._normalize_person_record(replacement)
            target_name = normalized["name"]

            if target_name in people:
                people[target_name] = normalized
                replaced += 1
                continue

            original_name = replacement.get("original_name")
            if not isinstance(original_name, str) or not original_name.strip():
                raise ValueError(
                    "Replacement for a renamed person must include a non-empty 'original_name'"
                )

            normalized_original_name = original_name.strip()
            if normalized_original_name not in people:
                raise ValueError(f"No person found with name: {normalized_original_name}")
            if target_name in people and target_name != normalized_original_name:
                raise ValueError(f"A person with name '{target_name}' already exists")

            del people[normalized_original_name]
            people[target_name] = normalized
            replaced += 1
            renamed += 1

        self._write_all(people)
        return replaced, renamed

    def _normalize_person_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)

        name = normalized.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Record must contain a non-empty 'name'")
        normalized["name"] = name.strip()

        affiliation = normalized.get("affiliation")
        if isinstance(affiliation, str):
            normalized["affiliation"] = affiliation.strip()

        normalized_flags = _normalize_flags(normalized.get("flags"))
        if normalized_flags:
            normalized["flags"] = normalized_flags
        else:
            normalized.pop("flags", None)

        normalized.pop("affiliations", None)
        normalized.pop("original_name", None)
        return normalized

    def _load_from_lines(self, lines: Iterable[str]) -> dict[str, dict[str, Any]]:
        people: dict[str, dict[str, Any]] = {}
        for line in lines:
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
                people[normalized_name] = _merge_person_record(
                    {},
                    {"name": normalized_name, **record},
                )
            else:
                people[normalized_name] = _merge_person_record(
                    existing,
                    {"name": normalized_name, **record},
                )

        return people

    def _people_repo_path(self) -> str:
        return self.path.relative_to(self.repo_dir).as_posix()

    def _read_people_file_from_commit(self, commit: str) -> str:
        people_filename = self._people_repo_path()
        show_result = self._git("show", f"{commit}:{people_filename}", check=False)
        if show_result.returncode == 0:
            return show_result.stdout

        stderr = show_result.stderr.strip()
        if "exists on disk, but not in" in stderr or "path '" in stderr and "does not exist in" in stderr:
            return ""
        raise RuntimeError(stderr or f"git show failed for commit {commit}")

    def _prepare_write_base(self, base_commit: str | None) -> None:
        if base_commit is None:
            return

        target_commit = self.resolve_history_commit(base_commit)
        head_commit = self._current_head_commit()
        if head_commit == target_commit:
            return

        self._git("reset", "--hard", target_commit)

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
            f"--work-tree={self.repo_dir}",
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

    def _current_head_commit(self) -> str | None:
        self._ensure_local_repo()
        rev_parse = self._git("rev-parse", "HEAD", check=False)
        if rev_parse.returncode == 0:
            return rev_parse.stdout.strip()

        stderr = rev_parse.stderr.strip()
        if "unknown revision or path not in the working tree" in stderr or "Needed a single revision" in stderr:
            return None
        raise RuntimeError(stderr or "git rev-parse failed")

    def _commit_people_file(self, people_count: int) -> None:
        self._ensure_local_repo()

        people_filename = self._people_repo_path()
        if self.path.parent != self.repo_dir:
            raise ValueError(
                f"People file path must be inside repository work tree {self.repo_dir}, got {self.path}"
            )

        self._git("add", "--", people_filename)

        diff_status = self._git("diff", "--cached", "--quiet", "--", people_filename, check=False)
        if diff_status.returncode == 0:
            return
        if diff_status.returncode != 1:
            raise RuntimeError(diff_status.stderr.strip() or "git diff failed")

        self._git("commit", "-m", f"Update people.jsonl ({people_count} people)")
