"""Utilities for reading and updating JSONL stores in the data repository.

This module is the single point of entry for writes to store-managed files
inside the local data git repository under ``data/.people_repo``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PEOPLE_REPO_DIRNAME = ".people_repo"
DEFAULT_PEOPLE_REPO_PATH = Path(__file__).resolve().parent.parent / "data" / DEFAULT_PEOPLE_REPO_DIRNAME
DEFAULT_PEOPLE_PATH = DEFAULT_PEOPLE_REPO_PATH / "people.jsonl"
DEFAULT_EXPERTISE_EMBEDDINGS_PATH = DEFAULT_PEOPLE_REPO_PATH / "expertise_embeddings.jsonl"
DEFAULT_PAPER_EXPERTISE_EMBEDDINGS_PATH = DEFAULT_PEOPLE_REPO_PATH / "paper_expertise_embeddings.jsonl"
DEFAULT_DBLP_FILTERED_PATH = DEFAULT_PEOPLE_REPO_PATH / "dblp_filtered.jsonl"
DEFAULT_PEOPLE_REPO_REMOTE = "https://github.com/michaelpradel/Chairvana-fse2027-data.git"
DEFAULT_PEOPLE_REPO_REMOTE_NAME = "origin"
DEFAULT_PEOPLE_REPO_BRANCH = "main"


class RemoteConflictError(RuntimeError):
    """Raised when a write is rejected because the remote data repository
    has advanced beyond the local state, indicating a concurrent write by
    another user."""


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

    # Publication summaries are recomputed as a whole and should replace the
    # previous summary instead of merging nested venue counts.
    if "publication_summary" in updates:
        merged["publication_summary"] = updates["publication_summary"]

    merged.pop("affiliations", None)
    return merged


class DataStore:
    """Read and update data repository records stored as JSONL."""

    def __init__(
        self,
        path: Path | None = None,
        repo_dir: Path | None = None,
        expertise_embeddings_path: Path | None = None,
        paper_expertise_embeddings_path: Path | None = None,
        remote_url: str = DEFAULT_PEOPLE_REPO_REMOTE,
        remote_name: str = DEFAULT_PEOPLE_REPO_REMOTE_NAME,
        push_branch: str = DEFAULT_PEOPLE_REPO_BRANCH,
        auto_push: bool = True,
        auto_sync_on_conflict: bool = True,
    ) -> None:
        self.path = path or DEFAULT_PEOPLE_PATH
        if repo_dir is not None:
            self.repo_dir = repo_dir
        elif self.path.parent.name == DEFAULT_PEOPLE_REPO_DIRNAME:
            self.repo_dir = self.path.parent
        else:
            self.repo_dir = self.path.parent / DEFAULT_PEOPLE_REPO_DIRNAME
        self.expertise_embeddings_path = expertise_embeddings_path or (
            self.repo_dir / DEFAULT_EXPERTISE_EMBEDDINGS_PATH.name
        )
        self.paper_expertise_embeddings_path = paper_expertise_embeddings_path or (
            self.repo_dir / DEFAULT_PAPER_EXPERTISE_EMBEDDINGS_PATH.name
        )
        self.dblp_filtered_path = self.repo_dir / DEFAULT_DBLP_FILTERED_PATH.name
        self.remote_url = remote_url
        self.remote_name = remote_name
        self.push_branch = push_branch
        self.auto_push = auto_push
        self.auto_sync_on_conflict = auto_sync_on_conflict
        self._force_push_on_next_commit = False

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

    def load_expertise_embeddings(self, commit: str | None = None) -> dict[str, dict[str, Any]]:
        if commit is None:
            if not self.expertise_embeddings_path.exists():
                return {}

            with self.expertise_embeddings_path.open("r", encoding="utf-8") as file_obj:
                return self._load_expertise_from_lines(file_obj)

        normalized_commit = self.resolve_history_commit(commit)
        content = self._read_file_from_commit(normalized_commit, self._expertise_repo_path())
        return self._load_expertise_from_lines(content.splitlines())

    def load_paper_expertise_embeddings(self, commit: str | None = None) -> dict[str, dict[str, Any]]:
        if commit is None:
            if not self.paper_expertise_embeddings_path.exists():
                return {}

            with self.paper_expertise_embeddings_path.open("r", encoding="utf-8") as file_obj:
                return self._load_expertise_from_lines(file_obj)

        normalized_commit = self.resolve_history_commit(commit)
        content = self._read_file_from_commit(normalized_commit, self._paper_expertise_repo_path())
        return self._load_expertise_from_lines(content.splitlines())

    def list_history(self) -> list[dict[str, str]]:
        self._ensure_local_repo()
        people_filename = self._people_repo_path()
        expertise_filename = self._expertise_repo_path()
        log_result = self._git(
            "log",
            "--format=%H%x1f%h%x1f%cs%x1f%s",
            "--",
            people_filename,
            expertise_filename,
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

    def update_many_expertise(
        self,
        entries: Iterable[dict[str, Any]],
        *,
        base_commit: str | None = None,
    ) -> tuple[int, int]:
        self._prepare_write_base(base_commit)
        embeddings_by_name = self.load_expertise_embeddings()
        added = 0
        updated = 0

        for entry in entries:
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Each expertise entry must contain a non-empty 'name': {entry}")

            normalized_name = name.strip()
            normalized_entry = {"name": normalized_name, **entry}

            if normalized_name in embeddings_by_name:
                embeddings_by_name[normalized_name] = _merge_dict(
                    embeddings_by_name[normalized_name],
                    normalized_entry,
                )
                updated += 1
            else:
                embeddings_by_name[normalized_name] = normalized_entry
                added += 1

        self._write_all_expertise(embeddings_by_name)
        return added, updated

    def update_many_paper_expertise(
        self,
        entries: Iterable[dict[str, Any]],
        *,
        base_commit: str | None = None,
    ) -> tuple[int, int]:
        self._prepare_write_base(base_commit)
        embeddings_by_name = self.load_paper_expertise_embeddings()
        added = 0
        updated = 0

        for entry in entries:
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Each paper expertise entry must contain a non-empty 'name': {entry}")

            normalized_name = name.strip()
            normalized_entry = {"name": normalized_name, **entry}

            if normalized_name in embeddings_by_name:
                embeddings_by_name[normalized_name] = _merge_dict(
                    embeddings_by_name[normalized_name],
                    normalized_entry,
                )
                updated += 1
            else:
                embeddings_by_name[normalized_name] = normalized_entry
                added += 1

        self._write_all_paper_expertise(embeddings_by_name)
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

    def add_person(
        self,
        record: dict[str, Any],
        *,
        base_commit: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        self._prepare_write_base(base_commit)
        people = self.load()

        normalized_record = self._normalize_person_record(record)
        new_name = normalized_record["name"]

        existing_names_casefold = {name.casefold() for name in people}
        if new_name.casefold() in existing_names_casefold:
            return False, people.get(new_name, normalized_record)

        people[new_name] = normalized_record
        self._write_all(people)
        return True, normalized_record

    def delete_person(
        self,
        name: str,
        *,
        base_commit: str | None = None,
    ) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name must be non-empty")

        self._prepare_write_base(base_commit)
        people = self.load()
        if normalized_name not in people:
            raise ValueError(f"No person found with name: {normalized_name}")

        del people[normalized_name]
        self._write_all(people)

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

    def overwrite_all(
        self,
        records: Iterable[dict[str, Any]],
        *,
        base_commit: str | None = None,
    ) -> int:
        self._prepare_write_base(base_commit)

        normalized_records: dict[str, dict[str, Any]] = {}
        for record in records:
            normalized = self._normalize_person_record(record)
            name = normalized["name"]
            if name in normalized_records:
                raise ValueError(f"Duplicate name in overwrite set: {name}")
            normalized_records[name] = normalized

        self._write_all(normalized_records)
        return len(normalized_records)

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

    def _load_expertise_from_lines(self, lines: Iterable[str]) -> dict[str, dict[str, Any]]:
        embeddings_by_name: dict[str, dict[str, Any]] = {}
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
            embeddings_by_name[normalized_name] = {"name": normalized_name, **record}

        return embeddings_by_name

    def _people_repo_path(self) -> str:
        return self.path.relative_to(self.repo_dir).as_posix()

    def _expertise_repo_path(self) -> str:
        return self.expertise_embeddings_path.relative_to(self.repo_dir).as_posix()

    def _paper_expertise_repo_path(self) -> str:
        return self.paper_expertise_embeddings_path.relative_to(self.repo_dir).as_posix()

    def _dblp_filtered_repo_path(self) -> str:
        return self.dblp_filtered_path.relative_to(self.repo_dir).as_posix()

    def replace_dblp_filtered(self, source_path: Path, *, base_commit: str | None = None) -> None:
        """Replace dblp_filtered.jsonl in the repo and commit the update."""
        if not source_path.exists():
            raise FileNotFoundError(f"Missing filtered DBLP JSONL source: {source_path}")

        self._prepare_write_base(base_commit)
        self.dblp_filtered_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, self.dblp_filtered_path)
        self._commit_store_files(
            message="Update dblp_filtered.jsonl",
            repo_paths=[self._dblp_filtered_repo_path()],
        )

    def write_dblp_filtered_lines(
        self,
        lines: Iterable[str],
        *,
        base_commit: str | None = None,
    ) -> None:
        """Write dblp_filtered.jsonl content and commit the update."""
        self._prepare_write_base(base_commit)
        self.dblp_filtered_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dblp_filtered_path.open("w", encoding="utf-8") as file_obj:
            for line in lines:
                file_obj.write(line)
        self._commit_store_files(
            message="Update dblp_filtered.jsonl",
            repo_paths=[self._dblp_filtered_repo_path()],
        )

    def _read_people_file_from_commit(self, commit: str) -> str:
        return self._read_file_from_commit(commit, self._people_repo_path())

    def _read_file_from_commit(self, commit: str, repo_filename: str) -> str:
        show_result = self._git("show", f"{commit}:{repo_filename}", check=False)
        if show_result.returncode == 0:
            return show_result.stdout

        stderr = show_result.stderr.strip()
        if "exists on disk, but not in" in stderr or "path '" in stderr and "does not exist in" in stderr:
            return ""
        raise RuntimeError(stderr or f"git show failed for commit {commit}")

    def _prepare_write_base(self, base_commit: str | None) -> None:
        self._force_push_on_next_commit = False
        self._check_remote_not_advanced()
        if base_commit is None:
            return

        target_commit = self.resolve_history_commit(base_commit)
        head_commit = self._current_head_commit()
        if head_commit == target_commit:
            return

        self._git("reset", "--hard", target_commit)
        # Historical writes intentionally rewrite branch history, so a
        # subsequent push may require force-with-lease.
        self._force_push_on_next_commit = True

    def _write_all(self, people: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sorted_names = sorted(people, key=str.casefold)

        with self.path.open("w", encoding="utf-8") as file_obj:
            for name in sorted_names:
                record = people[name]
                record.pop("affiliations", None)
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._commit_store_files(
            message=f"Update people.jsonl ({len(sorted_names)} people)",
            repo_paths=[self._people_repo_path()],
        )

    def _write_all_expertise(self, embeddings_by_name: dict[str, dict[str, Any]]) -> None:
        self.expertise_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_names = sorted(embeddings_by_name, key=str.casefold)

        with self.expertise_embeddings_path.open("w", encoding="utf-8") as file_obj:
            for name in sorted_names:
                record = embeddings_by_name[name]
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._commit_store_files(
            message=f"Update expertise_embeddings.jsonl ({len(sorted_names)} people)",
            repo_paths=[self._expertise_repo_path()],
        )

    def _write_all_paper_expertise(self, embeddings_by_name: dict[str, dict[str, Any]]) -> None:
        self.paper_expertise_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_names = sorted(embeddings_by_name, key=str.casefold)

        with self.paper_expertise_embeddings_path.open("w", encoding="utf-8") as file_obj:
            for name in sorted_names:
                record = embeddings_by_name[name]
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._commit_store_files(
            message=f"Update paper_expertise_embeddings.jsonl ({len(sorted_names)} papers)",
            repo_paths=[self._paper_expertise_repo_path()],
        )

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
            self._ensure_remote_configured()
            return

        self.repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(self.repo_dir)], check=True, capture_output=True, text=True)
        self._git("config", "user.name", "People Store Bot")
        self._git("config", "user.email", "people-store@local")
        self._ensure_remote_configured()

    def _ensure_remote_configured(self) -> None:
        if not self.remote_url:
            return

        remote_result = self._git("remote", "get-url", self.remote_name, check=False)
        if remote_result.returncode == 0:
            current_url = remote_result.stdout.strip()
            if current_url != self.remote_url:
                self._git("remote", "set-url", self.remote_name, self.remote_url)
            return

        stderr = remote_result.stderr.strip()
        if "No such remote" in stderr:
            self._git("remote", "add", self.remote_name, self.remote_url)
            return

        raise RuntimeError(stderr or f"git remote get-url failed for {self.remote_name}")

    def _fetch_remote(self) -> None:
        fetch_result = self._git("fetch", self.remote_name, check=False)
        if fetch_result.returncode != 0:
            stderr = fetch_result.stderr.strip()
            raise RuntimeError(stderr or f"git fetch {self.remote_name} failed")

    def _check_remote_not_advanced(self) -> None:
        if not self.auto_push or not self.remote_url:
            return
        git_dir = self.repo_dir / ".git"
        if not git_dir.exists():
            return  # Fresh repo; no remote state to compare against.
        self._fetch_remote()
        local_head = self._current_head_commit()
        if local_head is None:
            return  # No local commits yet; nothing to conflict with.
        remote_ref = f"{self.remote_name}/{self.push_branch}"
        remote_result = self._git("rev-parse", remote_ref, check=False)
        if remote_result.returncode != 0:
            return  # Remote branch does not exist yet; first push.
        remote_head = remote_result.stdout.strip()
        if local_head != remote_head:
            if self.auto_sync_on_conflict:
                # Automatically fast-forward local to remote HEAD so the
                # upcoming write is applied on top of the latest data.
                self._git("reset", "--hard", remote_head)
                return
            raise RemoteConflictError(
                "Data was modified by another user. Your edit was lost. "
                "Please reload before making any edits."
            )

    def _current_head_commit(self) -> str | None:
        self._ensure_local_repo()
        rev_parse = self._git("rev-parse", "HEAD", check=False)
        if rev_parse.returncode == 0:
            return rev_parse.stdout.strip()

        stderr = rev_parse.stderr.strip()
        if "unknown revision or path not in the working tree" in stderr or "Needed a single revision" in stderr:
            return None
        raise RuntimeError(stderr or "git rev-parse failed")

    def _commit_store_files(self, message: str, repo_paths: list[str]) -> None:
        self._ensure_local_repo()

        for repo_path in repo_paths:
            file_path = self.repo_dir / repo_path
            if file_path.parent != self.repo_dir:
                raise ValueError(
                    f"Store file path must be inside repository work tree {self.repo_dir}, got {file_path}"
                )

        self._git("add", "--", *repo_paths)

        diff_status = self._git("diff", "--cached", "--quiet", "--", *repo_paths, check=False)
        if diff_status.returncode == 0:
            return
        if diff_status.returncode != 1:
            raise RuntimeError(diff_status.stderr.strip() or "git diff failed")

        self._git("commit", "-m", message)
        try:
            if self.auto_push:
                self._push_latest_commit(force_with_lease=self._force_push_on_next_commit)
        finally:
            self._force_push_on_next_commit = False

    def _push_latest_commit(self, *, force_with_lease: bool) -> None:
        push_args = ["push"]
        if force_with_lease:
            push_args.append("--force-with-lease")
        push_args.extend([self.remote_name, f"HEAD:{self.push_branch}"])
        push_result = self._git(*push_args, check=False)
        if push_result.returncode == 0:
            return

        # Push rejected — another user pushed during the write window.
        # Roll back the local commit, then resync local HEAD to remote.
        self._git("reset", "--hard", "HEAD~1", check=False)
        fetch_result = self._git("fetch", self.remote_name, check=False)
        if fetch_result.returncode == 0:
            self._git(
                "reset",
                "--hard",
                f"{self.remote_name}/{self.push_branch}",
                check=False,
            )
        raise RemoteConflictError(
            "Data was modified by another user. Your edit was lost. "
            "Please reload before making any edits."
        )
