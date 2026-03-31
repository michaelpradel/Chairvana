from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Pricing source: https://developers.openai.com/api/docs/pricing
# Values are USD per 1M tokens.
_MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
}

_MODEL_ALIASES: dict[str, str] = {
    # Add aliases only when there is no explicit pricing entry.
}

_LOG_ENTRY_PATTERN = re.compile(
    r"Time:\s*(?P<time>[^\n]+)\n"
    r"Description:\s*(?P<description>[^\n]+)\n"
    r"Tokens:\s*(?P<input>\d+)\s+input,\s*(?P<output>\d+)\s+output\s+\((?P<total>\d+)\s+total\)",
    re.MULTILINE,
)


def strip_model_date_suffix(model: str) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)


def model_pricing(model: str) -> dict[str, Any]:
    normalized_model = strip_model_date_suffix(model)
    direct_match_model = model if model in _MODEL_PRICING_PER_1M else normalized_model if normalized_model in _MODEL_PRICING_PER_1M else None
    alias_model = _MODEL_ALIASES.get(model) or _MODEL_ALIASES.get(normalized_model)
    resolved_model = direct_match_model or alias_model or normalized_model
    pricing = _MODEL_PRICING_PER_1M.get(resolved_model)
    pricing_mode = "exact" if direct_match_model else "estimated" if alias_model and pricing else "unknown"

    return {
        "requested_model": model,
        "normalized_model": normalized_model,
        "resolved_model": resolved_model,
        "input_per_1m": pricing["input"] if pricing else None,
        "output_per_1m": pricing["output"] if pricing else None,
        "has_pricing": pricing is not None,
        "is_aliased": bool(alias_model),
        "pricing_mode": pricing_mode,
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
    }


def parse_log_timestamp(raw_value: str) -> datetime | None:
    value = raw_value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def load_llm_usage_records(logs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for log_file in sorted(logs_dir.glob("llm_*.log")):
        try:
            content = log_file.read_text(encoding="utf-8")
        except OSError:
            continue

        for match in _LOG_ENTRY_PATTERN.finditer(content):
            timestamp = parse_log_timestamp(match.group("time"))
            input_tokens = int(match.group("input"))
            output_tokens = int(match.group("output"))
            total_tokens = int(match.group("total"))
            records.append(
                {
                    "timestamp": timestamp,
                    "timestamp_raw": match.group("time").strip(),
                    "description": match.group("description").strip(),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "source_file": log_file.name,
                }
            )

    records.sort(
        key=lambda record: (
            record["timestamp"] is None,
            record["timestamp"] or datetime.min,
            record["source_file"],
        )
    )
    return records


def compute_token_cost(input_tokens: int, output_tokens: int, pricing: dict[str, Any]) -> float | None:
    input_rate = pricing.get("input_per_1m")
    output_rate = pricing.get("output_per_1m")
    if input_rate is None or output_rate is None:
        return None
    return (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate


def llm_usage_stats(model: str, logs_dir: Path) -> dict[str, Any]:
    pricing = model_pricing(model)
    records = load_llm_usage_records(logs_dir)

    totals = {
        "calls": len(records),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_available": pricing["has_pricing"],
    }

    timeline_labels: list[str] = []
    timeline_input: list[int] = []
    timeline_output: list[int] = []
    timeline_total: list[int] = []
    timeline_cost_cumulative: list[float] = []

    grouped: dict[str, dict[str, Any]] = {}

    cumulative_cost = 0.0
    for record in records:
        input_tokens = record["input_tokens"]
        output_tokens = record["output_tokens"]
        total_tokens = record["total_tokens"]

        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["total_tokens"] += total_tokens

        label = (
            record["timestamp"].strftime("%Y-%m-%d %H:%M")
            if record["timestamp"] is not None
            else record["timestamp_raw"]
        )
        timeline_labels.append(label)
        timeline_input.append(input_tokens)
        timeline_output.append(output_tokens)
        timeline_total.append(total_tokens)

        call_cost = compute_token_cost(input_tokens, output_tokens, pricing)
        if call_cost is not None:
            cumulative_cost += call_cost
        timeline_cost_cumulative.append(round(cumulative_cost, 2))

        description = record["description"] or "(no description)"
        bucket = grouped.setdefault(
            description,
            {
                "description": description,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_available": pricing["has_pricing"],
            },
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += total_tokens
        if call_cost is not None:
            bucket["cost_usd"] += call_cost

    if pricing["has_pricing"]:
        totals["cost_usd"] = round(cumulative_cost, 2)

    grouped_rows = sorted(grouped.values(), key=lambda row: row["total_tokens"], reverse=True)
    for row in grouped_rows:
        row["cost_usd"] = round(row["cost_usd"], 2)

    return {
        "pricing": pricing,
        "totals": totals,
        "timeline": {
            "labels": timeline_labels,
            "input_tokens": timeline_input,
            "output_tokens": timeline_output,
            "total_tokens": timeline_total,
            "cost_cumulative_usd": timeline_cost_cumulative,
        },
        "grouped_by_description": grouped_rows,
        "records": records,
    }
