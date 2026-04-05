"""Shared helpers for querying OpenAI models in Chairvana."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, TypeVar

from openai import OpenAI
from pydantic import BaseModel


DEFAULT_RESPONSES_MODEL = "gpt-5-mini-2025-08-07"

TModel = TypeVar("TModel", bound=BaseModel)

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def log_llm_call(
    description: str,
    prompt: str,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Append a structured entry to the daily LLM log file."""
    _LOGS_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = _LOGS_DIR / f"llm_{today}.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    total = input_tokens + output_tokens
    separator = "=" * 80
    entry = (
        f"\n{separator}\n"
        f"Time:        {now}\n"
        f"Description: {description}\n"
        f"Tokens:      {input_tokens} input, {output_tokens} output ({total} total)\n"
        f"--- Prompt ---\n{prompt}\n"
        f"--- Response ---\n{response_text}\n"
    )
    with log_file.open("a", encoding="utf-8") as f:
        f.write(entry)


def load_api_key(token_path: Path | None = None) -> str:
    resolved_token_path = token_path or (Path(__file__).resolve().parent.parent.parent / ".openai_token")
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
    description: str = "",
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

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    log_llm_call(
        description or "(no description)",
        input_text,
        result.model_dump_json(indent=2),
        input_tokens,
        output_tokens,
    )

    return result
