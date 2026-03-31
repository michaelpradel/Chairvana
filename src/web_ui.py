from __future__ import annotations

import argparse
import os
import re
import random
import threading
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import shlex
from typing import Any

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from clean_people import clean_single
from add_expertise_embeddings import cosine_similarity, get_or_create_topic_embedding, normalize_topic_text
from expertise_gap_finder import find_expertise_gaps
from llm_queries import DEFAULT_RESPONSES_MODEL
from data_store import DataStore, RemoteConflictError
from query_dblp import DblpQueryEngine, PACMPL_PREFIX, TARGET_VENUE_PREFIXES
from sync_people_with_publications import sync_single_person_publications


app = Flask(__name__)
app.config["SECRET_KEY"] = "chairvana-dev-key"

store = DataStore(auto_sync_on_conflict=False)

_clean_lock = threading.Lock()
_clean_status: dict[str, Any] = {
    "running": False,
    "error": None,
    "changed": False,
    "result_name": None,
}

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

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

TOPIC_MIN_SIMILARITY = 0.22
TOPIC_STRONG_SIMILARITY = 0.42
TOPIC_TOOLTIP_NEIGHBOR_COUNT = 3
TOPIC_TOOLTIP_MAX_YEARS_BACK = 5

SORT_MODE_RANDOM = "random"
SORT_MODE_ALPHA = "alpha"
_VALID_SORT_MODES = {SORT_MODE_RANDOM, SORT_MODE_ALPHA}

_GLOBAL_REGIONS = ["North America", "South America", "Europe", "Asia", "Africa"]

# Keep the region set fixed to five buckets for the Distribution panel.
# Oceania and Antarctica country codes are grouped into Asia and Africa, respectively.
_REGION_COUNTRY_CODES: dict[str, set[str]] = {
    "North America": {
        "AIA", "ATG", "ABW", "BHS", "BRB", "BLZ", "BMU", "BES", "VGB", "CAN", "CYM", "CRI", "CUB",
        "CUW", "DMA", "DOM", "SLV", "GRL", "GRD", "GLP", "GTM", "HTI", "HND", "JAM", "MTQ", "MEX",
        "MSR", "NIC", "PAN", "PRI", "BLM", "KNA", "LCA", "MAF", "SPM", "VCT", "SXM", "TTO", "TCA",
        "USA", "VIR", "UMI",
    },
    "South America": {
        "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "FLK", "GUF", "GUY", "PRY", "PER", "SGS", "SUR",
        "URY", "VEN",
    },
    "Europe": {
        "ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CZE", "DNK", "EST", "FRO", "FIN",
        "FRA", "DEU", "GIB", "GRC", "GGY", "VAT", "HUN", "ISL", "IRL", "IMN", "ITA", "JEY", "LVA",
        "LIE", "LTU", "LUX", "MLT", "MDA", "MCO", "MNE", "NLD", "MKD", "NOR", "POL", "PRT", "ROU",
        "RUS", "SMR", "SRB", "SVK", "SVN", "ESP", "SJM", "SWE", "CHE", "UKR", "GBR", "ALA",
    },
    "Asia": {
        "AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "CXR", "CCK", "CYP", "GEO",
        "HKG", "IND", "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "KWT", "KGZ", "LAO", "LBN",
        "MAC", "MYS", "MDV", "MNG", "MMR", "NPL", "PRK", "OMN", "PAK", "PSE", "PHL", "QAT", "SAU",
        "SGP", "KOR", "LKA", "SYR", "TWN", "TJK", "THA", "TLS", "TUR", "TKM", "ARE", "UZB", "VNM",
        "YEM", "IOT",
        # Oceania grouped into Asia to keep the required five-region chart.
        "ASM", "AUS", "COK", "FJI", "PYF", "GUM", "HMD", "KIR", "MHL", "FSM", "NRU", "NCL", "NZL",
        "NIU", "NFK", "MNP", "PLW", "PNG", "PCN", "WSM", "SLB", "TKL", "TON", "TUV", "VUT", "WLF",
    },
    "Africa": {
        "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD",
        "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN",
        "LSO", "LBR", "LBY", "MDG", "MWI", "MLI", "MRT", "MUS", "MYT", "MAR", "MOZ", "NAM", "NER",
        "NGA", "REU", "RWA", "SHN", "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA",
        "TGO", "TUN", "UGA", "ESH", "ZMB", "ZWE",
        # Antarctica grouped into Africa to keep the required five-region chart.
        "ATA",
    },
}

COUNTRY_CODE_TO_REGION: dict[str, str] = {
    country_code: region
    for region, country_codes in _REGION_COUNTRY_CODES.items()
    for country_code in country_codes
}

_NORMALIZED_REGION_NAMES: dict[str, str] = {
    re.sub(r"\s+", " ", region).strip().casefold(): region for region in _GLOBAL_REGIONS
}

_dblp_engine_lock = threading.Lock()
_dblp_engine: DblpQueryEngine | None = None


def _get_dblp_engine() -> DblpQueryEngine | None:
    global _dblp_engine
    if _dblp_engine is not None:
        return _dblp_engine

    with _dblp_engine_lock:
        if _dblp_engine is not None:
            return _dblp_engine
        try:
            _dblp_engine = DblpQueryEngine(preload_index=True)
        except Exception:  # noqa: BLE001
            return None
    return _dblp_engine


def _is_target_publication_key(key: str) -> bool:
    normalized_key = key.strip().casefold()
    if not normalized_key:
        return False
    return any(
        normalized_key == prefix or normalized_key.startswith(f"{prefix}/")
        for prefix in TARGET_VENUE_PREFIXES
    )


def _is_oopsla_pacmpl_key(key: str, issue: str | None) -> bool:
    normalized_key = key.strip().casefold()
    if not normalized_key.startswith(f"{PACMPL_PREFIX}/"):
        return False
    normalized_issue = (issue or "").strip().upper()
    return normalized_issue.startswith("OOPSLA")


def _nearest_topic_papers_for_person(
    person_name: str,
    topic_vectors: list[list[float]],
    paper_embeddings_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not topic_vectors:
        return []

    engine = _get_dblp_engine()
    if engine is None:
        return []

    try:
        publications = engine.query_author(person_name)
    except Exception:  # noqa: BLE001
        return []

    min_year = datetime.now().year - TOPIC_TOOLTIP_MAX_YEARS_BACK
    scored_papers: list[dict[str, Any]] = []
    seen_record_names: set[str] = set()

    for publication in publications:
        if not isinstance(publication.year, int) or publication.year < min_year:
            continue
        if not isinstance(publication.key, str) or not publication.key.strip():
            continue

        key = publication.key.strip()
        if not (_is_target_publication_key(key) or _is_oopsla_pacmpl_key(key, publication.issue)):
            continue

        record_name = f"paper:{key}"
        if record_name in seen_record_names:
            continue
        seen_record_names.add(record_name)

        record = paper_embeddings_by_name.get(record_name)
        if not isinstance(record, dict):
            continue

        paper_vector = _clean_embedding_vector(record.get("embedding"))
        if paper_vector is None:
            continue

        similarities = [
            similarity
            for similarity in (cosine_similarity(paper_vector, topic_vector) for topic_vector in topic_vectors)
            if similarity is not None
        ]
        if not similarities:
            continue

        average_similarity = sum(similarities) / len(similarities)

        raw_title = record.get("title")
        if not isinstance(raw_title, str) or not raw_title.strip():
            raw_title = publication.title if isinstance(publication.title, str) and publication.title.strip() else "Untitled"

        record_year = record.get("year")
        year = record_year if isinstance(record_year, int) else publication.year

        scored_papers.append(
            {
                "title": " ".join(raw_title.split()),
                "year": year,
                "similarity": round(average_similarity, 4),
            }
        )

    scored_papers.sort(
        key=lambda paper: (
            -float(paper.get("similarity", 0.0)),
            str(paper.get("title", "")).casefold(),
        )
    )
    return scored_papers[:TOPIC_TOOLTIP_NEIGHBOR_COUNT]


def _build_topic_tooltip(topic_similarity: float, nearest_papers: list[dict[str, Any]]) -> str:
    lines = [f"Topic similarity: {topic_similarity:.3f}"]
    if nearest_papers:
        lines.append("Top matching papers:")
        for index, paper in enumerate(nearest_papers, start=1):
            title = str(paper.get("title", "Untitled")).strip() or "Untitled"
            year = paper.get("year")
            year_suffix = f" ({year})" if isinstance(year, int) else ""
            similarity = float(paper.get("similarity", 0.0))
            lines.append(f"{index}. {title}{year_suffix} [sim {similarity:.3f}]")
    else:
        lines.append("Top matching papers: unavailable")
    return "\n".join(lines)


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


def _normalize_region_name(raw_region: str) -> str | None:
    normalized = re.sub(r"\s+", " ", raw_region).strip().casefold()
    if not normalized:
        return None
    return _NORMALIZED_REGION_NAMES.get(normalized)


def _region_for_person(person: dict[str, Any]) -> str | None:
    country = person.get("country")
    if not isinstance(country, str) or not country.strip():
        return None
    return COUNTRY_CODE_TO_REGION.get(country.strip().upper())


def _strip_model_date_suffix(model: str) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)


def _model_pricing(model: str) -> dict[str, Any]:
    normalized_model = _strip_model_date_suffix(model)
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


def _parse_log_timestamp(raw_value: str) -> datetime | None:
    value = raw_value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _load_llm_usage_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for log_file in sorted(_LOGS_DIR.glob("llm_*.log")):
        try:
            content = log_file.read_text(encoding="utf-8")
        except OSError:
            continue

        for match in _LOG_ENTRY_PATTERN.finditer(content):
            timestamp = _parse_log_timestamp(match.group("time"))
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


def _compute_token_cost(input_tokens: int, output_tokens: int, pricing: dict[str, Any]) -> float | None:
    input_rate = pricing.get("input_per_1m")
    output_rate = pricing.get("output_per_1m")
    if input_rate is None or output_rate is None:
        return None
    return (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate


def _llm_usage_stats(model: str) -> dict[str, Any]:
    pricing = _model_pricing(model)
    records = _load_llm_usage_records()

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

        call_cost = _compute_token_cost(input_tokens, output_tokens, pricing)
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


def _tagged_people_distribution(commit: str | None, tag_query: str) -> dict[str, Any]:
    selected_tags = _normalize_tag_query(tag_query)
    all_people = store.list_people(commit=commit)

    if not selected_tags:
        # No tag given: show distribution of the entire set of people
        matched_people = all_people
    else:
        matched_people = []
        for person in all_people:
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
    region_counter: Counter[str] = Counter()

    for person in matched_people:
        gender = person.get("gender")
        if isinstance(gender, str) and gender.strip():
            gender_counter[gender.strip().casefold()] += 1
        else:
            gender_counter["unknown"] += 1

        country = person.get("country")
        if isinstance(country, str) and country.strip():
            normalized_country = country.strip().upper()
            country_counter[normalized_country] += 1
            region = _region_for_person(person)
            if region is not None:
                region_counter[region] += 1
        else:
            country_counter["UNKNOWN"] += 1

    return {
        "input": " ".join(selected_tags),
        "selected_tags": selected_tags,
        "matched_count": len(matched_people),
        "gender": dict(gender_counter.most_common()),
        "country": dict(country_counter.most_common(12)),
        "region": {
            region: region_counter[region]
            for region in _GLOBAL_REGIONS
            if region_counter[region] > 0
        },
    }


def _top_common_tags(people: list[dict[str, Any]], limit: int = 5) -> list[str]:
    tag_counter: Counter[str] = Counter()

    for person in people:
        flags = person.get("flags")
        if not isinstance(flags, list):
            continue

        # Count each tag at most once per person to avoid skew from duplicates.
        unique_tags_for_person: set[str] = set()
        for flag in flags:
            if not isinstance(flag, str):
                continue
            normalized = flag.strip().casefold()
            if not normalized:
                continue
            if not normalized.startswith("#"):
                normalized = f"#{normalized}"
            if normalized == "#":
                continue
            unique_tags_for_person.add(normalized)

        tag_counter.update(unique_tags_for_person)

    return [tag for tag, _ in tag_counter.most_common(limit)]


def _parse_search_query(query: str) -> dict[str, Any]:
    """
    Parse a search query into structured filters.
    Supports:
    - Text search: matches name, affiliation, country, region, flags
    - Tag absence: "-#invite" or "-invite"
    - Gender: "male", "female", or "gender:unknown"
    - Country: "country:USA", "country:DEU", or "country:unknown"
    - Region: "region:Europe" or 'region:"North America"'
    - Affiliation presence: "affiliation:unknown"
    - Publications: "pubs>5", "pubs<=10", etc.
    - PC memberships: "pcs>3", "pcs<5", etc.
    - Topic embeddings: "t:quantum" or 't:"compiler testing"'

    Multiple filters can be separated by spaces or commas.
    """
    import re

    filters: dict[str, Any] = {
        "text": [],
        "topics": [],
        "exclude_tags": [],
        "gender": None,
        "country": None,
        "region": None,
        "affiliation_state": None,
        "pubs_op": None,
        "pubs_count": None,
        "pcs_op": None,
        "pcs_count": None,
    }

    # Treat commas as separators unless they appear in quoted strings.
    query_normalized = query.replace(",", " ")
    try:
        tokens = shlex.split(query_normalized)
    except ValueError:
        # Fallback for malformed quotes.
        tokens = query_normalized.strip().split()

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        lowered_token = token.casefold()

        # Exclude people who have the given tag, e.g., -#invite or -invite.
        if re.match(r"^-#?[a-z0-9_\-]+$", lowered_token):
            raw_tag = lowered_token[1:]
            normalized_tag = raw_tag if raw_tag.startswith("#") else f"#{raw_tag}"
            if normalized_tag != "#" and normalized_tag not in filters["exclude_tags"]:
                filters["exclude_tags"].append(normalized_tag)
            continue

        # Semantic topic filter (quoted phrases supported via shlex, e.g., t:"compiler testing").
        if lowered_token.startswith("t:"):
            topic_term = normalize_topic_text(token[2:])
            if topic_term:
                filters["topics"].append(topic_term)
            continue

        # Check for gender
        if lowered_token in ("male", "female"):
            filters["gender"] = lowered_token
        elif match := re.match(r"^gender:(male|female|unknown)$", token, re.IGNORECASE):
            filters["gender"] = match.group(1).casefold()
        elif match := re.match(r"^country:(.+)$", token, re.IGNORECASE):
            raw_country = match.group(1).strip()
            filters["country"] = "unknown" if raw_country.casefold() == "unknown" else raw_country.upper()
        elif match := re.match(r"^region:(.+)$", token, re.IGNORECASE):
            raw_region = match.group(1).strip()
            filters["region"] = "unknown" if raw_region.casefold() == "unknown" else _normalize_region_name(raw_region)
        elif match := re.match(r"^affiliation:(known|unknown)$", token, re.IGNORECASE):
            filters["affiliation_state"] = match.group(1).casefold()
        elif lowered_token == "unknown-affiliation":
            filters["affiliation_state"] = "unknown"
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
            filters["text"].append(lowered_token)

    return filters


def _matches_tag_filter(person: dict[str, Any], selected_tags: list[str]) -> bool:
    if not selected_tags:
        return True

    flags = person.get("flags")
    if not isinstance(flags, list):
        return False

    normalized_flags = {
        str(flag).strip().casefold() for flag in flags if isinstance(flag, str) and str(flag).strip()
    }
    return any(tag in normalized_flags for tag in selected_tags)


def _clean_embedding_vector(raw_embedding: Any) -> list[float] | None:
    if not isinstance(raw_embedding, list) or len(raw_embedding) < 2:
        return None

    vector: list[float] = []
    for value in raw_embedding:
        if not isinstance(value, (int, float)):
            return None
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
        vector.append(numeric_value)
    return vector


def _dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(a * b for a, b in zip(lhs, rhs, strict=True))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _normalize_vector(vector: list[float]) -> list[float] | None:
    norm = _vector_norm(vector)
    if norm <= 1e-12:
        return None
    return [value / norm for value in vector]


def _matvec_cov(centered_vectors: list[list[float]], vector: list[float]) -> list[float]:
    sample_count = len(centered_vectors)
    if sample_count <= 1:
        return [0.0 for _ in vector]

    scalar_projections = [_dot(row, vector) for row in centered_vectors]
    result = [0.0 for _ in vector]
    for row, projection in zip(centered_vectors, scalar_projections, strict=True):
        for index, value in enumerate(row):
            result[index] += projection * value

    scale = 1.0 / (sample_count - 1)
    return [value * scale for value in result]


def _principal_component(
    centered_vectors: list[list[float]],
    basis_vectors: list[list[float]],
    max_iterations: int = 12,
) -> list[float] | None:
    if not centered_vectors:
        return None

    dimension = len(centered_vectors[0])
    candidate = centered_vectors[0][:]
    if _vector_norm(candidate) <= 1e-12:
        candidate = [1.0 for _ in range(dimension)]

    for base in basis_vectors:
        projection = _dot(candidate, base)
        candidate = [value - projection * base_value for value, base_value in zip(candidate, base, strict=True)]

    normalized_candidate = _normalize_vector(candidate)
    if normalized_candidate is None:
        return None

    candidate = normalized_candidate
    for _ in range(max_iterations):
        next_candidate = _matvec_cov(centered_vectors, candidate)
        for base in basis_vectors:
            projection = _dot(next_candidate, base)
            next_candidate = [
                value - projection * base_value
                for value, base_value in zip(next_candidate, base, strict=True)
            ]

        normalized_next = _normalize_vector(next_candidate)
        if normalized_next is None:
            return None

        delta = _vector_norm([
            a - b for a, b in zip(normalized_next, candidate, strict=True)
        ])
        candidate = normalized_next
        if delta <= 1e-6:
            break

    return candidate


def _project_embeddings_2d(embeddings: list[list[float]]) -> list[tuple[float, float]]:
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [(0.0, 0.0)]

    dimension = len(embeddings[0])
    means = [0.0 for _ in range(dimension)]
    for vector in embeddings:
        for index, value in enumerate(vector):
            means[index] += value
    sample_count = len(embeddings)
    means = [value / sample_count for value in means]

    centered = [[value - means[index] for index, value in enumerate(vector)] for vector in embeddings]

    first_component = _principal_component(centered, basis_vectors=[])
    if first_component is None:
        return [(0.0, 0.0) for _ in embeddings]

    second_component = _principal_component(centered, basis_vectors=[first_component])
    if second_component is None:
        second_component = [0.0 for _ in range(dimension)]
        if dimension > 1:
            second_component[1] = 1.0

    xs = [_dot(vector, first_component) for vector in centered]
    ys = [_dot(vector, second_component) for vector in centered]

    x_scale = max((abs(value) for value in xs), default=1.0)
    y_scale = max((abs(value) for value in ys), default=1.0)
    x_scale = x_scale if x_scale > 1e-9 else 1.0
    y_scale = y_scale if y_scale > 1e-9 else 1.0

    return [(x / x_scale, y / y_scale) for x, y in zip(xs, ys, strict=True)]


def _matches_search_filters(person: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check if a person matches all the parsed search filters."""
    flags = person.get("flags")
    person_region = _region_for_person(person)
    normalized_flags: set[str] = set()
    if isinstance(flags, list):
        normalized_flags = {
            (flag.strip().casefold() if flag.strip().startswith("#") else f"#{flag.strip().casefold()}")
            for flag in flags
            if isinstance(flag, str) and flag.strip()
        }

    # Check absent-tag filter.
    if filters["exclude_tags"]:
        if any(tag in normalized_flags for tag in filters["exclude_tags"]):
            return False

    # Check gender filter
    if filters["gender"] is not None:
        raw_gender = person.get("gender")
        person_gender = raw_gender.strip().casefold() if isinstance(raw_gender, str) and raw_gender.strip() else "unknown"
        if person_gender != filters["gender"]:
            return False

    # Check country filter
    if filters["country"] is not None:
        raw_country = person.get("country")
        person_country = raw_country.strip().upper() if isinstance(raw_country, str) and raw_country.strip() else "unknown"
        if person_country != filters["country"]:
            return False

    # Check region filter
    if filters["region"] is not None:
        normalized_region = person_region if person_region is not None else "unknown"
        if normalized_region != filters["region"]:
            return False

    # Check affiliation presence filter
    if filters["affiliation_state"] is not None:
        raw_affiliation = person.get("affiliation")
        has_affiliation = isinstance(raw_affiliation, str) and bool(raw_affiliation.strip())
        if filters["affiliation_state"] == "unknown" and has_affiliation:
            return False
        if filters["affiliation_state"] == "known" and not has_affiliation:
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
        region = person_region.casefold() if isinstance(person_region, str) else ""
        flags_text = " ".join(flags).casefold() if isinstance(flags, list) else ""

        searchable_text = f"{name} {affiliation} {country} {region} {flags_text}"

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


def _normalize_country_code(raw_country: str) -> str:
    normalized = raw_country.strip().upper()
    if not normalized:
        return ""
    if not re.fullmatch(r"[A-Z]{3}", normalized):
        raise ValueError("Country must be a 3-letter code like DEU.")
    return normalized


def _normalize_sort_mode(raw_sort_mode: str) -> str:
    sort_mode = raw_sort_mode.strip().casefold()
    if sort_mode in _VALID_SORT_MODES:
        return sort_mode
    return SORT_MODE_RANDOM


def _sorted_people(people: list[dict[str, Any]], sort_mode: str) -> list[dict[str, Any]]:
    if sort_mode == SORT_MODE_ALPHA:
        return sorted(people, key=lambda person: str(person.get("name", "")).casefold())

    shuffled_people = list(people)
    random.shuffle(shuffled_people)
    return shuffled_people


def _filtered_people(query: str, sort_mode: str, commit: str | None = None) -> tuple[list[dict[str, Any]], int]:
    people = store.list_people(commit=commit)
    total_count = len(people)

    if not query.strip():
        return _sorted_people(people, sort_mode), total_count

    filters = _parse_search_query(query)

    filtered: list[dict[str, Any]] = []
    for person in people:
        if _matches_search_filters(person, filters):
            filtered.append(person)

    topics = filters.get("topics") or []
    if topics:
        embeddings_by_name = store.load_expertise_embeddings(commit=commit)
        paper_embeddings_by_name = store.load_paper_expertise_embeddings(commit=commit)

        topic_vectors: list[list[float]] = []
        for topic in topics:
            topic_vector: list[float] | None = None
            topic_record_key = f"__topic__:{topic}"
            cached_topic_record = embeddings_by_name.get(topic_record_key)
            if isinstance(cached_topic_record, dict):
                raw_cached_vector = cached_topic_record.get("embedding")
                if isinstance(raw_cached_vector, list) and all(isinstance(value, (int, float)) for value in raw_cached_vector):
                    topic_vector = [float(value) for value in raw_cached_vector]

            # Create and cache topic embedding on first query.
            if topic_vector is None:
                try:
                    created_topic = get_or_create_topic_embedding(store, topic)
                except Exception:  # noqa: BLE001
                    created_topic = None
                if isinstance(created_topic, dict):
                    raw_created_vector = created_topic.get("embedding")
                    if isinstance(raw_created_vector, list) and all(
                        isinstance(value, (int, float)) for value in raw_created_vector
                    ):
                        topic_vector = [float(value) for value in raw_created_vector]

            if topic_vector is not None:
                topic_vectors.append(topic_vector)

        if topic_vectors:
            semantic_matches: list[dict[str, Any]] = []
            for person in filtered:
                name = person.get("name")
                if not isinstance(name, str):
                    continue

                person_embedding_record = embeddings_by_name.get(name)
                if not isinstance(person_embedding_record, dict):
                    continue
                raw_person_vector = person_embedding_record.get("embedding")
                if not isinstance(raw_person_vector, list) or not all(
                    isinstance(value, (int, float)) for value in raw_person_vector
                ):
                    continue

                person_vector = [float(value) for value in raw_person_vector]

                topic_scores: list[float] = []
                for topic_vector in topic_vectors:
                    score = cosine_similarity(person_vector, topic_vector)
                    if score is not None:
                        topic_scores.append(score)

                if not topic_scores:
                    continue

                average_score = sum(topic_scores) / len(topic_scores)
                if average_score < TOPIC_MIN_SIMILARITY:
                    continue

                if average_score >= TOPIC_STRONG_SIMILARITY:
                    font_alpha = 1.0
                else:
                    ratio = (average_score - TOPIC_MIN_SIMILARITY) / (
                        TOPIC_STRONG_SIMILARITY - TOPIC_MIN_SIMILARITY
                    )
                    clamped_ratio = max(0.0, min(1.0, ratio))
                    font_alpha = 0.3 + 0.7 * clamped_ratio

                enriched_person = dict(person)
                enriched_person["_topic_similarity"] = round(average_score, 4)
                enriched_person["_topic_font_alpha"] = round(font_alpha, 3)
                nearest_papers = _nearest_topic_papers_for_person(name, topic_vectors, paper_embeddings_by_name)
                enriched_person["_topic_paper_neighbors"] = nearest_papers
                enriched_person["_topic_tooltip"] = _build_topic_tooltip(average_score, nearest_papers)
                semantic_matches.append(enriched_person)

            semantic_matches.sort(
                key=lambda person: (
                    -float(person.get("_topic_similarity", 0.0)),
                    str(person.get("name", "")).casefold(),
                )
            )
            filtered = semantic_matches
        else:
            filtered = []

    return _sorted_people(filtered, sort_mode), total_count


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


def _request_wants_json() -> bool:
    requested_with = request.headers.get("X-Requested-With", "")
    if requested_with.casefold() == "xmlhttprequest":
        return True

    best = request.accept_mimetypes.best_match(["application/json", "text/html"])
    return best == "application/json" and (
        request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]
    )


def _person_publication_count(person: dict[str, Any]) -> int:
    pub_summary = person.get("publication_summary")
    if not isinstance(pub_summary, dict):
        return 0

    total = pub_summary.get("total")
    if isinstance(total, int) and total >= 0:
        return total
    return 0


def _person_prior_pc_count(person: dict[str, Any]) -> int:
    pc_memberships = person.get("pc_memberships")
    if not isinstance(pc_memberships, list):
        return 0
    return sum(1 for membership in pc_memberships if isinstance(membership, dict))


def _enrich_people_for_search_list(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched_people: list[dict[str, Any]] = []
    for person in people:
        enriched_person = dict(person)
        enriched_person["_publication_count"] = _person_publication_count(person)
        enriched_person["_prior_pc_count"] = _person_prior_pc_count(person)
        enriched_people.append(enriched_person)
    return enriched_people


def _build_index_context(
    *,
    query: str,
    sort_mode: str,
    selected_name: str,
    add_person_mode: bool,
    requested_history: str,
    dist_tags: str,
    expertise_tags: str,
    expertise_add: str,
    gap_tag: str,
    gap_year_start: str,
    gap_year_end: str,
    gap_min_similarity: str,
) -> dict[str, Any]:
    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        flash(str(exc), "error")
        history_state = store.get_history_state()

    active_commit = None if history_state["is_head"] else history_state["current_commit"]

    people, total_people_count = _filtered_people(query, sort_mode, active_commit)
    people = _enrich_people_for_search_list(people)
    matched_people_count = len(people)
    all_people = store.list_people(commit=active_commit)
    top_common_tags = _top_common_tags(all_people)
    distribution = {
        "input": " ".join(_normalize_tag_query(dist_tags)),
        "selected_tags": _normalize_tag_query(dist_tags),
        "matched_count": 0,
        "gender": {},
        "country": {},
        "region": {},
        "loading": True,
    }

    selected_person: dict[str, Any] | None = None
    if selected_name and not add_person_mode:
        for person in people:
            if person.get("name") == selected_name:
                selected_person = person
                break

    if selected_person is None and people and not add_person_mode:
        selected_person = people[0]

    with _clean_lock:
        clean_running = _clean_status["running"]

    return {
        "people": people,
        "top_common_tags": top_common_tags,
        "selected_person": selected_person,
        "add_person_mode": add_person_mode,
        "publication_display_counts": _publication_display_counts(selected_person),
        "selected_name": selected_name,
        "query": query,
        "sort_mode": sort_mode,
        "total_people_count": total_people_count,
        "matched_people_count": matched_people_count,
        "clean_running": clean_running,
        "history_commit": history_state["current_commit"],
        "history_entry": history_state["current_entry"],
        "history_is_head": history_state["is_head"],
        "older_history_commit": history_state["older_commit"],
        "newer_history_commit": history_state["newer_commit"],
        "distribution": distribution,
        "expertise_tags": expertise_tags,
        "expertise_add": expertise_add,
        "gap_tag": gap_tag,
        "gap_year_start": gap_year_start,
        "gap_year_end": gap_year_end,
        "gap_min_similarity": gap_min_similarity,
        "all_people_names": [person.get("name", "") for person in all_people if person.get("name")],
    }


@app.get("/")
def index() -> str:
    query = request.args.get("q", "")
    sort_mode = _normalize_sort_mode(request.args.get("sort", SORT_MODE_RANDOM))
    selected_name = request.args.get("selected", "").strip()
    add_person_mode = request.args.get("add_person", "").strip().casefold() in {"1", "true", "yes"}
    requested_history = request.args.get("history", "").strip()
    dist_tags = request.args.get("dist_tags", "#invite").strip()
    expertise_tags = request.args.get("expertise_tags", dist_tags).strip()
    expertise_add = request.args.get("expertise_add", "").strip()
    gap_tag = request.args.get("gap_tag", "#invite").strip() or "#invite"
    gap_year_start = request.args.get("gap_year_start", "2024").strip() or "2024"
    gap_year_end = request.args.get("gap_year_end", "2026").strip() or "2026"
    gap_min_similarity = request.args.get("gap_min_similarity", "0.7").strip() or "0.7"

    context = _build_index_context(
        query=query,
        sort_mode=sort_mode,
        selected_name=selected_name,
        add_person_mode=add_person_mode,
        requested_history=requested_history,
        dist_tags=dist_tags,
        expertise_tags=expertise_tags,
        expertise_add=expertise_add,
        gap_tag=gap_tag,
        gap_year_start=gap_year_start,
        gap_year_end=gap_year_end,
        gap_min_similarity=gap_min_similarity,
    )
    return render_template("index.html", **context)


@app.get("/api/panels")
def panels_data() -> Any:
    query = request.args.get("q", "")
    sort_mode = _normalize_sort_mode(request.args.get("sort", SORT_MODE_RANDOM))
    selected_name = request.args.get("selected", "").strip()
    add_person_mode = request.args.get("add_person", "").strip().casefold() in {"1", "true", "yes"}
    requested_history = request.args.get("history", "").strip()
    dist_tags = request.args.get("dist_tags", "#invite").strip()
    expertise_tags = request.args.get("expertise_tags", dist_tags).strip()
    expertise_add = request.args.get("expertise_add", "").strip()
    gap_tag = request.args.get("gap_tag", "#invite").strip() or "#invite"
    gap_year_start = request.args.get("gap_year_start", "2024").strip() or "2024"
    gap_year_end = request.args.get("gap_year_end", "2026").strip() or "2026"
    gap_min_similarity = request.args.get("gap_min_similarity", "0.7").strip() or "0.7"

    context = _build_index_context(
        query=query,
        sort_mode=sort_mode,
        selected_name=selected_name,
        add_person_mode=add_person_mode,
        requested_history=requested_history,
        dist_tags=dist_tags,
        expertise_tags=expertise_tags,
        expertise_add=expertise_add,
        gap_tag=gap_tag,
        gap_year_start=gap_year_start,
        gap_year_end=gap_year_end,
        gap_min_similarity=gap_min_similarity,
    )

    return jsonify(
        {
            "list_html": render_template("_list_panel.html", **context),
            "edit_html": render_template("_edit_panel.html", **context),
            "selected_name": context["selected_person"]["name"] if context.get("selected_person") else "",
        }
    )


@app.post("/people/add-tag")
def add_tag_to_filtered_people() -> Any:
    query = request.form.get("q", "")
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()
    selected_name = request.form.get("selected", "").strip()
    raw_bulk_tag = request.form.get("bulk_tag", "").strip()

    normalized_tags = _normalize_tag_query(raw_bulk_tag)
    if len(normalized_tags) != 1:
        flash("Please provide exactly one tag (for example: #invite).", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=selected_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot apply tags while affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    selected=selected_name,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )

    filtered_people, _ = _filtered_people(query, sort_mode, history_commit)
    filtered_names = [
        person_name
        for person in filtered_people
        for person_name in [person.get("name")]
        if isinstance(person_name, str) and person_name.strip()
    ]

    if not filtered_names:
        flash("No people match the current filter; no tags were added.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=selected_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    tag = normalized_tags[0]
    updates = [{"name": name, "flags": [tag]} for name in filtered_names]

    try:
        _, updated_count = store.update_many(updates, base_commit=history_commit)
    except RemoteConflictError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=selected_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=selected_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    flash(f"Added {tag} to {updated_count} people from the current list.", "success")
    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            selected=selected_name,
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


@app.get("/api/distribution")
def distribution_data() -> Any:
    requested_history = request.args.get("history", "").strip()
    tags_input = request.args.get("tags", "").strip()

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    active_commit = None if history_state["is_head"] else history_state["current_commit"]
    distribution = _tagged_people_distribution(active_commit, tags_input)
    return jsonify(distribution)


@app.post("/person/create")
def create_person() -> Any:
    new_name = request.form.get("name", "").strip()
    new_affiliation = request.form.get("affiliation", "").strip()
    new_homepage = request.form.get("homepage", "").strip()
    new_gender = request.form.get("gender", "").strip()
    new_country = request.form.get("country", "").strip()
    flags = request.form.get("flags", "").strip()
    query = request.form.get("q", "")
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    try:
        normalized_country = _normalize_country_code(new_country)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    if not new_name:
        flash("Name is required for new people.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot create: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    add_person=1,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )

    existing_people = store.list_people(commit=history_commit)
    existing_names_casefold = {
        name.casefold()
        for person in existing_people
        for name in [person.get("name")]
        if isinstance(name, str) and name.strip()
    }
    if new_name.casefold() in existing_names_casefold:
        flash(f"A person with name '{new_name}' already exists.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    try:
        is_new, created = store.add_person(
            {
                "name": new_name,
                "affiliation": new_affiliation,
                "homepage": new_homepage,
                "flags": flags,
                "gender": new_gender.casefold() if new_gender else "",
                "country": normalized_country,
            },
            base_commit=history_commit,
        )
    except RemoteConflictError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    if not is_new:
        flash(f"A person with name '{new_name}' already exists.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                add_person=1,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    flash("Person added successfully.", "success")
    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            selected=created["name"],
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


@app.get("/api/expertise")
def expertise_projection() -> Any:
    requested_history = request.args.get("history", "").strip()
    tags_input = request.args.get("tags", "").strip()
    add_people_input = request.args.get("add_people", "").strip()
    add_topics_input = request.args.get("add_topics", "").strip()
    
    # Legacy support for old add_person parameter
    add_person_name = request.args.get("add_person", "").strip()

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    selected_tags = _normalize_tag_query(tags_input)
    people = store.list_people(commit=history_state["current_commit"])
    people_by_name = {
        person["name"]: person
        for person in people
        if isinstance(person, dict) and isinstance(person.get("name"), str) and person.get("name", "").strip()
    }

    base_people = [person for person in people if _matches_tag_filter(person, selected_tags)]
    base_names = {person.get("name") for person in base_people if isinstance(person.get("name"), str)}
    selected_names = {person["name"] for person in base_people if isinstance(person.get("name"), str)}

    # Parse added people from new parameter format
    added_people_list: list[str] = []
    if add_people_input:
        added_people_list = [name.strip() for name in add_people_input.split("||") if name.strip()]
    elif add_person_name:
        # Legacy support
        added_people_list = [add_person_name]

    # Add each person (case-insensitive lookup)
    for add_name in added_people_list:
        additional_person = people_by_name.get(add_name)
        if additional_person is None:
            folded_target = add_name.casefold()
            for person_name, person in people_by_name.items():
                if person_name.casefold() == folded_target:
                    additional_person = person
                    add_name = person_name
                    break
        if additional_person is not None and add_name not in selected_names:
            selected_names.add(add_name)

    embeddings_by_name = store.load_expertise_embeddings(commit=history_state["current_commit"])

    # Build list of selected people in sorted order
    selected_people: list[dict[str, Any]] = []
    person_point_type: dict[str, str] = {}  # Track whether each person is base or added
    
    for name in sorted(selected_names, key=str.casefold):
        person = people_by_name.get(name)
        if person is not None:
            selected_people.append(person)
            is_in_base = name in base_names
            person_point_type[name] = "base" if is_in_base else "added"

    vectors: list[list[float]] = []
    point_meta: list[dict[str, Any]] = []
    missing_embeddings: list[str] = []

    # Add people vectors
    for person in selected_people:
        name = person.get("name")
        if not isinstance(name, str):
            continue

        embedding_record = embeddings_by_name.get(name, {})
        vector = _clean_embedding_vector(embedding_record.get("embedding"))
        if vector is None:
            missing_embeddings.append(name)
            continue

        vectors.append(vector)
        point_meta.append(
            {
                "name": name,
                "point_type": person_point_type.get(name, "base"),
                "publication_count": embedding_record.get("publication_count")
                if isinstance(embedding_record.get("publication_count"), int)
                else None,
            }
        )

    # Add topic vectors
    added_topics_list: list[str] = []
    if add_topics_input:
        added_topics_list = [topic.strip() for topic in add_topics_input.split("||") if topic.strip()]

    for topic_query in added_topics_list:
        # Parse the topic query (e.g., "t:testing" or t:"compiler testing")
        if topic_query.lower().startswith("t:"):
            topic_term = normalize_topic_text(topic_query[2:])
        else:
            topic_term = normalize_topic_text(topic_query)

        if not topic_term:
            continue

        topic_vector: list[float] | None = None
        topic_record_key = f"__topic__:{topic_term}"
        
        # Check cache first
        cached_topic_record = embeddings_by_name.get(topic_record_key)
        if isinstance(cached_topic_record, dict):
            raw_cached_vector = cached_topic_record.get("embedding")
            if isinstance(raw_cached_vector, list) and all(isinstance(value, (int, float)) for value in raw_cached_vector):
                topic_vector = [float(value) for value in raw_cached_vector]

        # Create and cache topic embedding on first query
        if topic_vector is None:
            try:
                created_topic = get_or_create_topic_embedding(store, topic_term)
            except Exception:  # noqa: BLE001
                created_topic = None
            if isinstance(created_topic, dict):
                raw_created_vector = created_topic.get("embedding")
                if isinstance(raw_created_vector, list) and all(
                    isinstance(value, (int, float)) for value in raw_created_vector
                ):
                    topic_vector = [float(value) for value in raw_created_vector]

        if topic_vector is not None:
            vectors.append(topic_vector)
            point_meta.append(
                {
                    "name": f"Topic: {topic_term}",
                    "point_type": "added",
                    "publication_count": None,
                }
            )

    projected = _project_embeddings_2d(vectors)
    points = []
    for (x, y), meta in zip(projected, point_meta, strict=True):
        points.append(
            {
                **meta,
                "x": round(x, 6),
                "y": round(y, 6),
            }
        )

    return jsonify(
        {
            "tags": selected_tags,
            "base_count": len(base_people),
            "selected_count": len(selected_people),
            "with_embeddings": len(points),
            "without_embeddings": len(missing_embeddings),
            "missing_embedding_names": missing_embeddings[:30],
            "added_people": added_people_list,
            "added_topics": added_topics_list,
            "points": points,
        }
    )


@app.get("/api/expertise-gaps")
def expertise_gaps() -> Any:
    requested_history = request.args.get("history", "").strip()
    tag = request.args.get("tag", "#invite").strip() or "#invite"

    try:
        min_year = int(request.args.get("min_year", "2024"))
        max_year = int(request.args.get("max_year", "2026"))
        min_similarity = float(request.args.get("min_similarity", "0.7"))
    except ValueError:
        return jsonify({"error": "min_year, max_year, and min_similarity must be numeric"}), 400

    if min_year > max_year:
        return jsonify({"error": "min_year must be <= max_year"}), 400
    if min_similarity < -1.0 or min_similarity > 1.0:
        return jsonify({"error": "min_similarity must be between -1 and 1"}), 400

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    active_commit = None if history_state["is_head"] else history_state["current_commit"]

    try:
        result = find_expertise_gaps(
            tag=tag,
            min_year=min_year,
            max_year=max_year,
            min_similarity=min_similarity,
            top_k=5,
            max_results=300,
            base_commit=active_commit,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"failed to compute expertise gaps: {exc}"}), 500

    return jsonify(result)


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
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    try:
        normalized_country = _normalize_country_code(new_country)
    except ValueError as exc:
        if _request_wants_json():
            return jsonify({"error": str(exc)}), 400
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    if not original_name:
        if _request_wants_json():
            return jsonify({"error": "Missing original person name."}), 400
        flash("Missing original person name.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            if _request_wants_json():
                return jsonify({"error": "Cannot save: affiliation cleaning is in progress."}), 409
            flash("Cannot save: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    selected=original_name,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )

    try:
        updates: dict[str, Any] = {
            "name": new_name,
            "affiliation": new_affiliation,
            "homepage": new_homepage,
            "flags": flags,
            "gender": new_gender.casefold() if new_gender else "",
            "country": normalized_country,
        }
        updated = store.update_person(
            original_name,
            updates,
            base_commit=history_commit,
        )
    except RemoteConflictError as exc:
        if _request_wants_json():
            return jsonify({"error": str(exc), "conflict": True}), 409
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )
    except ValueError as exc:
        if _request_wants_json():
            return jsonify({"error": str(exc)}), 400
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    if _request_wants_json():
        return jsonify(
            {
                "message": "Person updated successfully.",
                "person": updated,
                "publication_display_counts": _publication_display_counts(updated),
                "selected_name": updated["name"],
                "history_is_head": True,
            }
        )

    flash("Person updated successfully.", "success")
    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            selected=updated["name"],
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


@app.post("/person/clean")
def clean_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cleaning is already in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    selected=original_name,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )
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
    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            selected=original_name,
            history=history_commit,
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


@app.get("/person/clean/status")
def get_clean_status() -> Any:
    with _clean_lock:
        return jsonify(dict(_clean_status))


@app.get("/llm-usage")
def llm_usage() -> str:
    stats = _llm_usage_stats(DEFAULT_RESPONSES_MODEL)
    return render_template(
        "llm_usage.html",
        model=DEFAULT_RESPONSES_MODEL,
        stats=stats,
    )


@app.post("/person/delete")
def delete_person() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot delete: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    selected=original_name,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )

    try:
        store.delete_person(
            original_name,
            base_commit=history_commit,
        )
    except RemoteConflictError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    flash(f"Person '{original_name}' deleted successfully.", "success")
    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


@app.post("/person/sync-publications")
def sync_person_publications() -> Any:
    original_name = request.form.get("original_name", "").strip()
    query = request.form.get("q", "")
    sort_mode = _normalize_sort_mode(request.form.get("sort", SORT_MODE_RANDOM))
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invite").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot auto-complete publications: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
                    sort=sort_mode,
                    selected=original_name,
                    history=history_commit,
                    dist_tags=dist_tags,
                    expertise_tags=expertise_tags,
                    expertise_add=expertise_add,
                )
            )

    try:
        changed, summary = sync_single_person_publications(
            original_name,
            base_commit=history_commit,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )
    except Exception as exc:  # noqa: BLE001
        flash(f"Failed to auto-complete publications: {exc}", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                sort=sort_mode,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    if summary is None:
        flash("No target-venue publications found for this person.", "error")
    elif changed:
        flash("Publication summary auto-completed successfully.", "success")
    else:
        flash("Publication summary already up to date.", "success")

    return redirect(
        url_for(
            "index",
            q=query,
            sort=sort_mode,
            selected=original_name,
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Chairvana web UI.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "5000")),
        help="Port for the web server (default: 5000 or PORT env var).",
    )
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    app.run(debug=True, port=args.port)
