from __future__ import annotations

import threading
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


def _filtered_people(query: str, commit: str | None = None) -> list[dict[str, Any]]:
    people = store.list_people(commit=commit)
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return people

    filtered: list[dict[str, Any]] = []
    for person in people:
        name = str(person.get("name", ""))
        affiliation = str(person.get("affiliation", ""))
        flags = person.get("flags")
        flags_text = " ".join(flags) if isinstance(flags, list) else ""

        if (
            normalized_query in name.casefold()
            or normalized_query in affiliation.casefold()
            or normalized_query in flags_text.casefold()
        ):
            filtered.append(person)
    return filtered


@app.get("/")
def index() -> str:
    query = request.args.get("q", "")
    selected_name = request.args.get("selected", "").strip()
    requested_history = request.args.get("history", "").strip()

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        flash(str(exc), "error")
        history_state = store.get_history_state()

    people = _filtered_people(query, history_state["current_commit"])

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
        selected_name=selected_name,
        query=query,
        clean_running=clean_running,
        history_commit=history_state["current_commit"],
        history_entry=history_state["current_entry"],
        history_is_head=history_state["is_head"],
        older_history_commit=history_state["older_commit"],
        newer_history_commit=history_state["newer_commit"],
    )


@app.post("/person/update")
def update_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    new_name = request.form.get("name", "").strip()
    new_affiliation = request.form.get("affiliation", "").strip()
    flags = request.form.get("flags", "").strip()
    query = request.form.get("q", "")
    history_commit = request.form.get("history", "").strip() or None

    if not original_name:
        flash("Missing original person name.", "error")
        return redirect(url_for("index", q=query, history=history_commit))

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot save: affiliation cleaning is in progress.", "error")
            return redirect(url_for("index", q=query, selected=original_name, history=history_commit))

    try:
        updated = store.update_person(
            original_name,
            {
                "name": new_name,
                "affiliation": new_affiliation,
                "flags": flags,
            },
            base_commit=history_commit,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index", q=query, selected=original_name, history=history_commit))

    flash("Person updated successfully.", "success")
    return redirect(url_for("index", q=query, selected=updated["name"]))


@app.post("/person/clean")
def clean_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    history_commit = request.form.get("history", "").strip() or None

    if not original_name:
        flash("No person selected.", "error")
        return redirect(url_for("index", q=query, history=history_commit))

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cleaning is already in progress.", "error")
            return redirect(url_for("index", q=query, selected=original_name, history=history_commit))
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
    return redirect(url_for("index", q=query, selected=original_name, history=history_commit))


@app.get("/person/clean/status")
def get_clean_status() -> Any:
    with _clean_lock:
        return jsonify(dict(_clean_status))


if __name__ == "__main__":
    app.run(debug=True)
