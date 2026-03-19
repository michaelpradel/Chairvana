"""Shared helpers for querying OpenAI models in Chairvana."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypeVar

from openai import OpenAI
from pydantic import BaseModel


DEFAULT_RESPONSES_MODEL = "gpt-5-mini-2025-08-07"

TModel = TypeVar("TModel", bound=BaseModel)


def load_api_key(token_path: Path | None = None) -> str:
    resolved_token_path = token_path or (Path(__file__).resolve().parent.parent / ".openai_token")
    try:
        token = resolved_token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing API token file: {resolved_token_path}") from exc

    if not token:
        raise ValueError(f"API token file is empty: {resolved_token_path}")

    return token


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=load_api_key())


def parse_structured_response(
    *,
    input_text: str,
    response_model: type[TModel],
    model: str = DEFAULT_RESPONSES_MODEL,
    tools: Iterable[Any] | None = None,
    client: OpenAI | None = None,
) -> TModel:
    active_client = client or get_openai_client()

    request_args: dict[str, Any] = {
        "model": model,
        "input": input_text,
        "text_format": response_model,
    }
    if tools is not None:
        request_args["tools"] = list(tools)

    response = active_client.responses.parse(**request_args)

    result = response.output_parsed
    if result is None:
        raise ValueError("Model response did not contain structured output")

    return result
