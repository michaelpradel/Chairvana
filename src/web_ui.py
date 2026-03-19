from __future__ import annotations

from typing import Any

from flask import Flask, flash, redirect, render_template, request, url_for

from people import PeopleStore


app = Flask(__name__)
app.config["SECRET_KEY"] = "chairvana-dev-key"

store = PeopleStore()


def _filtered_people(query: str) -> list[dict[str, Any]]:
    people = store.list_people()
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return people

    filtered: list[dict[str, Any]] = []
    for person in people:
        name = str(person.get("name", ""))
        affiliation = str(person.get("affiliation", ""))
        if normalized_query in name.casefold() or normalized_query in affiliation.casefold():
            filtered.append(person)
    return filtered


@app.get("/")
def index() -> str:
    query = request.args.get("q", "")
    selected_name = request.args.get("selected", "").strip()

    people = _filtered_people(query)

    selected_person: dict[str, Any] | None = None
    if selected_name:
        for person in people:
            if person.get("name") == selected_name:
                selected_person = person
                break

    if selected_person is None and people:
        selected_person = people[0]

    return render_template(
        "index.html",
        people=people,
        selected_person=selected_person,
        query=query,
    )


@app.post("/person/update")
def update_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    new_name = request.form.get("name", "").strip()
    new_affiliation = request.form.get("affiliation", "").strip()
    query = request.form.get("q", "")

    if not original_name:
        flash("Missing original person name.", "error")
        return redirect(url_for("index", q=query))

    try:
        updated = store.update_person(
            original_name,
            {
                "name": new_name,
                "affiliation": new_affiliation,
            },
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index", q=query, selected=original_name))

    flash("Person updated successfully.", "success")
    return redirect(url_for("index", q=query, selected=updated["name"]))


if __name__ == "__main__":
    app.run(debug=True)
