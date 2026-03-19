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

from openai import OpenAI
from pydantic import BaseModel, HttpUrl, TypeAdapter


MODEL = "gpt-5-mini-2025-08-07"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "select_best_web_result.txt"
HOMEPAGE_EMAIL_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "select_homepage_and_email.txt"
)


class SearchResult(BaseModel):
    top_link: str


class HomepageAndEmailResult(BaseModel):
    homepage: str
    email: str


URL_ADAPTER = TypeAdapter(HttpUrl)


def load_api_key() -> str:
    token_path = Path(__file__).resolve().parent.parent / ".openai_token"
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing API token file: {token_path}") from exc

    if not token:
        raise ValueError(f"API token file is empty: {token_path}")

    return token


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
    client = OpenAI(api_key=load_api_key())

    response = client.responses.parse(
        model=MODEL,
        input=load_prompt(query),
        tools=[{"type": "web_search"}],
        text_format=SearchResult,
    )

    result = response.output_parsed
    if result is None:
        raise ValueError("Model response did not contain structured output")

    top_link = str(URL_ADAPTER.validate_python(result.top_link))
    return SearchResult(top_link=top_link)


def find_homepage_and_email(person: str) -> HomepageAndEmailResult:
    """Return a researcher's personal homepage and email address."""
    client = OpenAI(api_key=load_api_key())

    response = client.responses.parse(
        model=MODEL,
        input=load_homepage_email_prompt(person),
        tools=[{"type": "web_search"}],
        text_format=HomepageAndEmailResult,
    )

    result = response.output_parsed
    if result is None:
        raise ValueError("Model response did not contain structured output")

    homepage = str(URL_ADAPTER.validate_python(result.homepage))
    email = deobfuscate_email(result.email)
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
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc
