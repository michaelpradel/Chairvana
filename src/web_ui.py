from __future__ import annotations

import re
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


def _top_common_tags(people: list[dict[str, Any]], limit: int = 3) -> list[str]:
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
    - Text search: matches name, affiliation, country, flags
    - Gender: "male", "female", or "gender:unknown"
    - Country: "country:USA", "country:DEU", or "country:unknown"
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
        "gender": None,
        "country": None,
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

    topics = filters.get("topics") or []
    if topics:
        embeddings_by_name = store.load_expertise_embeddings(commit=commit)

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
    expertise_tags = request.args.get("expertise_tags", dist_tags).strip()
    expertise_add = request.args.get("expertise_add", "").strip()

    try:
        history_state = store.get_history_state(requested_history or None)
    except ValueError as exc:
        flash(str(exc), "error")
        history_state = store.get_history_state()

    active_commit = None if history_state["is_head"] else history_state["current_commit"]

    people, total_people_count = _filtered_people(query, active_commit)
    matched_people_count = len(people)
    all_people = store.list_people(commit=active_commit)
    distribution = _tagged_people_distribution(active_commit, dist_tags)

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
        top_common_tags=_top_common_tags(people),
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
        expertise_tags=expertise_tags,
        expertise_add=expertise_add,
        all_people_names=[person.get("name", "") for person in all_people if person.get("name")],
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
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("Missing original person name.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    with _clean_lock:
        if _clean_status["running"]:
            flash("Cannot save: affiliation cleaning is in progress.", "error")
            return redirect(
                url_for(
                    "index",
                    q=query,
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
            "country": new_country.upper() if new_country else "",
        }
        updated = store.update_person(
            original_name,
            updates,
            base_commit=history_commit,
        )
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
                selected=original_name,
                history=history_commit,
                dist_tags=dist_tags,
                expertise_tags=expertise_tags,
                expertise_add=expertise_add,
            )
        )

    flash("Person updated successfully.", "success")
    return redirect(
        url_for(
            "index",
            q=query,
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
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invited").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
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
    history_commit = request.form.get("history", "").strip() or None
    dist_tags = request.form.get("dist_tags", "#invited").strip()
    expertise_tags = request.form.get("expertise_tags", dist_tags).strip()
    expertise_add = request.form.get("expertise_add", "").strip()

    if not original_name:
        flash("No person selected.", "error")
        return redirect(
            url_for(
                "index",
                q=query,
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
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(
            url_for(
                "index",
                q=query,
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
            dist_tags=dist_tags,
            expertise_tags=expertise_tags,
            expertise_add=expertise_add,
        )
    )


if __name__ == "__main__":
    app.run(debug=True)
