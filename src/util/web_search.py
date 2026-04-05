"""Simple web search example using the OpenAI Responses API.

Reads an API key from `.openai_token`, loads a prompt template from
`prompts/`, and asks GPT-5 Mini to use web search to select the single
best matching page for a query.

This module can be imported and used from other Python scripts or run as a CLI.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

from util.llm_queries import DEFAULT_RESPONSES_MODEL, parse_structured_response


MODEL = DEFAULT_RESPONSES_MODEL
PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "select_best_web_result.txt"
HOMEPAGE_EMAIL_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "select_homepage_and_email.txt"
)
RESEARCH_TRACK_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "find_research_track_pc_page.txt"
)


class SearchResult(BaseModel):
    top_link: str


class HomepageAndEmailResult(BaseModel):
    homepage: str
    email: str


URL_ADAPTER = TypeAdapter(HttpUrl)


def normalize_optional_homepage(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""

    try:
        return str(URL_ADAPTER.validate_python(candidate))
    except ValidationError:
        print(f"[Web][Warn] Invalid homepage returned by LLM: {value!r}")
        return ""


def load_prompt(query: str) -> str:
    try:
        prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {PROMPT_PATH}") from exc

    return prompt_template.format(query=query)


def load_homepage_email_prompt(person: str) -> str:
    try:
        prompt_template = HOMEPAGE_EMAIL_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing prompt template: {HOMEPAGE_EMAIL_PROMPT_PATH}"
        ) from exc

    return prompt_template.format(person=person)


def deobfuscate_email(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith("mailto:"):
        lowered = lowered[len("mailto:") :]

    replacements: list[tuple[str, str]] = [
        (r"\s*\(at\)\s*", "@"),
        (r"\s*\[at\]\s*", "@"),
        (r"\s*\{at\}\s*", "@"),
        (r"\s+-at-\s+", "@"),
        (r"\s+at\s+", "@"),
        (r"\s*\(dot\)\s*", "."),
        (r"\s*\[dot\]\s*", "."),
        (r"\s*\{dot\}\s*", "."),
        (r"\s+-dot-\s+", "."),
        (r"\s+dot\s+", "."),
    ]

    normalized = lowered
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    normalized = re.sub(r"\s+", "", normalized)

    candidates = re.findall(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", normalized)
    if not candidates:
        raise ValueError(f"Could not parse a valid email from: {value}")

    return candidates[0]


def search_best_web_result(query: str) -> SearchResult:
    """Return the best matching web link for the given query."""
    result = parse_structured_response(
        input_text=load_prompt(query),
        response_model=SearchResult,
        description="search best web result",
        model=MODEL,
        tools=[{"type": "web_search"}],
    )

    top_link = str(URL_ADAPTER.validate_python(result.top_link))
    return SearchResult(top_link=top_link)


def search_research_track_pc_page(conference: str, year: int) -> SearchResult:
    """Return the researchr.org URL for the main research track PC page of the given conference/year."""
    try:
        prompt_template = RESEARCH_TRACK_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing prompt template: {RESEARCH_TRACK_PROMPT_PATH}") from exc

    prompt = prompt_template.format(conference=conference, year=year)
    result = parse_structured_response(
        input_text=prompt,
        response_model=SearchResult,
        description="search research track PC page",
        model=MODEL,
        tools=[{"type": "web_search"}],
    )
    top_link = str(URL_ADAPTER.validate_python(result.top_link))
    return SearchResult(top_link=top_link)


def find_homepage_and_email(person: str) -> HomepageAndEmailResult:
    """Return a researcher's personal homepage and email address."""
    result = parse_structured_response(
        input_text=load_homepage_email_prompt(person),
        response_model=HomepageAndEmailResult,
        description="find homepage and email",
        model=MODEL,
        tools=[{"type": "web_search"}],
    )

    homepage = normalize_optional_homepage(result.homepage)
    email_candidate = result.email.strip()
    if not email_candidate:
        email = ""
    else:
        try:
            email = deobfuscate_email(email_candidate)
        except ValueError:
            print(f"[Web][Warn] Invalid email returned by LLM: {result.email!r}")
            email = ""

    return HomepageAndEmailResult(homepage=homepage, email=email)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best web result for a search query.")
    parser.add_argument("query", help="Search query to run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = search_best_web_result(args.query)
    print(json.dumps({"top_link": result.top_link}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
