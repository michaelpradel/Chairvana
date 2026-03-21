from __future__ import annotations

import threading
from collections import Counter
from typing import Any

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from clean_people import clean_single
from llm_queries import DEFAULT_RESPONSES_MODEL
from people import PeopleStore


app = Flask(__name__)
app.config["SECRET_KEY"] = "chairvana-dev-key"

store = PeopleStore()

_clean_lock = threading.Lock()
_clean_status: dict[str, Any] = {
    "running": False,
    "error": None,
    "changed": False,
    "result_name": None,
}


def _normalize_tag_query(raw_tags: str) -> list[str]:
    tokens = [token.strip() for token in raw_tags.replace(",", " ").split() if token.strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.casefold()
        if not lowered.startswith("#"):
            lowered = f"#{lowered}"
        if lowered == "#" or lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(lowered)
    return normalized


def _tagged_people_distribution(commit: str | None, tag_query: str) -> dict[str, Any]:
    selected_tags = _normalize_tag_query(tag_query)
    if not selected_tags:
        selected_tags = ["#invited"]

    matched_people: list[dict[str, Any]] = []
    for person in store.list_people(commit=commit):
        flags = person.get("flags")
        if not isinstance(flags, list):
            continue
        normalized_flags = {
            str(flag).strip().casefold() for flag in flags if isinstance(flag, str) and str(flag).strip()
        }
        if any(tag in normalized_flags for tag in selected_tags):
            matched_people.append(person)

    gender_counter: Counter[str] = Counter()
    country_counter: Counter[str] = Counter()

    for person in matched_people:
        gender = person.get("gender")
        if isinstance(gender, str) and gender.strip():
            gender_counter[gender.strip().casefold()] += 1
        else:
            gender_counter["unknown"] += 1

        country = person.get("country")
        if isinstance(country, str) and country.strip():
            country_counter[country.strip().upper()] += 1
        else:
            country_counter["UNKNOWN"] += 1

    return {
        "input": " ".join(selected_tags),
        "selected_tags": selected_tags,
        "matched_count": len(matched_people),
        "gender": dict(gender_counter.most_common()),
        "country": dict(country_counter.most_common(12)),
    }


def _parse_search_query(query: str) -> dict[str, Any]:
    """
    Parse a search query into structured filters.
    Supports:
    - Text search: matches name, affiliation, country, flags
    - Gender: "male" or "female"
    - Publications: "pubs>5", "pubs<=10", etc.
    - PC memberships: "pcs>3", "pcs<5", etc.
    
    Multiple filters can be separated by spaces or commas.
    """
    import re

    filters: dict[str, Any] = {
        "text": [],
        "gender": None,
        "pubs_op": None,
        "pubs_count": None,
        "pcs_op": None,
        "pcs_count": None,
    }

    # Treat commas as separators (convert to spaces) for more flexible input
    query_normalized = query.replace(",", " ")
    tokens = query_normalized.strip().split()
    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Check for gender
        if token.casefold() in ("male", "female"):
            filters["gender"] = token.casefold()
        # Check for publications filter (pubs>5, pubs<=10, etc.)
        elif match := re.match(r"^pubs(>=|<=|>|<|=)(\d+)$", token, re.IGNORECASE):
            filters["pubs_op"] = match.group(1)
            filters["pubs_count"] = int(match.group(2))
        # Check for PC memberships filter (pcs>3, pcs<5, etc.)
        elif match := re.match(r"^pcs(>=|<=|>|<|=)(\d+)$", token, re.IGNORECASE):
            filters["pcs_op"] = match.group(1)
            filters["pcs_count"] = int(match.group(2))
        # Otherwise, it's text search
        else:
            filters["text"].append(token.casefold())

    return filters


def _matches_search_filters(person: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check if a person matches all the parsed search filters."""
    # Check gender filter
    if filters["gender"] is not None:
        person_gender = str(person.get("gender", "")).casefold()
        if person_gender != filters["gender"]:
            return False

    # Check publication count filter
    if filters["pubs_op"] is not None and filters["pubs_count"] is not None:
        pub_summary = person.get("publication_summary")
        if not isinstance(pub_summary, dict):
            pub_total = 0
        else:
            pub_total = pub_summary.get("total", 0)

        if not _check_numeric_condition(pub_total, filters["pubs_op"], filters["pubs_count"]):
            return False

    # Check PC memberships count filter
    if filters["pcs_op"] is not None and filters["pcs_count"] is not None:
        pc_memberships = person.get("pc_memberships")
        pc_count = len(pc_memberships) if isinstance(pc_memberships, list) else 0

        if not _check_numeric_condition(pc_count, filters["pcs_op"], filters["pcs_count"]):
            return False

    # Check text search (must match at least one text token)
    if filters["text"]:
        name = str(person.get("name", "")).casefold()
        affiliation = str(person.get("affiliation", "")).casefold()
        country = str(person.get("country", "")).casefold()
        flags = person.get("flags")
        flags_text = " ".join(flags).casefold() if isinstance(flags, list) else ""

        searchable_text = f"{name} {affiliation} {country} {flags_text}"

        text_matched = any(text_token in searchable_text for text_token in filters["text"])
        if not text_matched:
            return False

    return True


def _check_numeric_condition(value: int, op: str, threshold: int) -> bool:
    """Check if a numeric value satisfies the given condition."""
    if op == ">":
        return value > threshold
    elif op == ">=":
        return value >= threshold
    elif op == "<":
        return value < threshold
    elif op == "<=":
        return value <= threshold
    elif op == "=":
        return value == threshold
    return False


def _filtered_people(query: str, commit: str | None = None) -> tuple[list[dict[str, Any]], int]:
    people = store.list_people(commit=commit)
    total_count = len(people)

    if not query.strip():
        return people, total_count

    filters = _parse_search_query(query)

    filtered: list[dict[str, Any]] = []
    for person in people:
        if _matches_search_filters(person, filters):
            filtered.append(person)

    return filtered, total_count


def _publication_display_counts(person: dict[str, Any] | None) -> list[tuple[str, int]]:
    if not isinstance(person, dict):
        return []

    pub_summary = person.get("publication_summary")
    if not isinstance(pub_summary, dict):
        return []

    raw_counts = pub_summary.get("by_venue")
    if not isinstance(raw_counts, dict):
        return []

    venue_labels = {
        "icse": "ICSE",
        "kbse": "ASE",
        "sigsoft": "FSE",
        "issta": "ISSTA",
        "oopsla": "OOPSLA",
        "pacmpl": "OOPSLA",
    }
    display_order = ["ICSE", "ASE", "FSE", "ISSTA", "OOPSLA"]

    aggregated: Counter[str] = Counter()
    for venue_key, count in raw_counts.items():
        if not isinstance(venue_key, str) or not isinstance(count, int):
            continue
        label = venue_labels.get(venue_key.casefold(), venue_key.upper())
        aggregated[label] += count

    return sorted(aggregated.items(), key=lambda item: (display_order.index(item[0]) if item[0] in display_order else len(display_order), item[0]))


@app.get("/")
def index() -> str:
    query = request.args.get("q", "")
    selected_name = request.args.get("selected", "").strip()
    requested_history = request.args.get("history", "").strip()
    dist_tags = request.args.get("dist_tags", "#invited").strip()

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        flash(str(exc), "error")
        history_state = store.get_history_state()

    people, total_people_count = _filtered_people(query, history_state["current_commit"])
    matched_people_count = len(people)
    distribution = _tagged_people_distribution(history_state["current_commit"], dist_tags)

    selected_person: dict[str, Any] | None = None
    if selected_name:
        for person in people:
            if person.get("name") == selected_name:
                selected_person = person
                break

    if selected_person is None and people:
        selected_person = people[0]

    with _clean_lock:
        clean_running = _clean_status["running"]

    return render_template(
        "index.html",
        people=people,
        selected_person=selected_person,
        publication_display_counts=_publication_display_counts(selected_person),
        selected_name=selected_name,
        query=query,
        total_people_count=total_people_count,
        matched_people_count=matched_people_count,
        clean_running=clean_running,
        history_commit=history_state["current_commit"],
        history_entry=history_state["current_entry"],
        history_is_head=history_state["is_head"],
        older_history_commit=history_state["older_commit"],
        newer_history_commit=history_state["newer_commit"],
        distribution=distribution,
    )


@app.post("/person/update")
def update_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    new_name = request.form.get("name", "").strip()
    new_affiliation = request.form.get("affiliation", "").strip()
    new_homepage = request.form.get("homepage", "").strip()
    new_gender = request.form.get("gender", "").strip()
    new_country = request.form.get("country", "").strip()
    flags = request.form.get("flags", "").strip()
    query = request.form.get("q", "")
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invited").strip()

    if not original_name:
        flash("Missing original person name.", "error")
        return redirect(url_for("index", q=query, history=history_commit, dist_tags=dist_tags))

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot save: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags)
            )

    try:
        updates: dict[str, Any] = {
            "name": new_name,
            "affiliation": new_affiliation,
            "homepage": new_homepage,
            "flags": flags,
            "gender": new_gender.casefold() if new_gender else "",
            "country": new_country.upper() if new_country else "",
        }
        updated = store.update_person(
            original_name,
            updates,
            base_commit=history_commit,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags))

    flash("Person updated successfully.", "success")
    return redirect(url_for("index", q=query, selected=updated["name"], dist_tags=dist_tags))


@app.post("/person/clean")
def clean_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invited").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(url_for("index", q=query, history=history_commit, dist_tags=dist_tags))

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cleaning is already in progress.", "error")
            return redirect(url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags))
        _clean_status.update({"running": True, "error": None, "changed": False, "result_name": None})

    def _run() -> None:
        try:
            changed, result_name = clean_single(
                store,
                original_name,
                DEFAULT_RESPONSES_MODEL,
                base_commit=history_commit,
            )
            with _clean_lock:
                _clean_status["changed"] = changed
                _clean_status["result_name"] = result_name
        except Exception as exc:  # noqa: BLE001
            with _clean_lock:
                _clean_status["error"] = str(exc)
        finally:
            with _clean_lock:
                _clean_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return redirect(url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags))


@app.get("/person/clean/status")
def get_clean_status() -> Any:
    with _clean_lock:
        return jsonify(dict(_clean_status))


@app.post("/person/delete")
def delete_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invited").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(url_for("index", q=query, history=history_commit, dist_tags=dist_tags))

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot delete: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags)
            )

    try:
        store.delete_person(
            original_name,
            base_commit=history_commit,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index", q=query, selected=original_name, history=history_commit, dist_tags=dist_tags))

    flash(f"Person '{original_name}' deleted successfully.", "success")
    return redirect(url_for("index", q=query, dist_tags=dist_tags))


if __name__ == "__main__":
    app.run(debug=True)
