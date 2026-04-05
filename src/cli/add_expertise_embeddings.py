"""Compute and store expertise embeddings for people.

This script computes one embedding per person from recent DBLP publications,
using the same target-venue filtering used by sync_people_with_publications.
Embeddings are written to expertise_embeddings.jsonl, while people.jsonl remains
human-readable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to allow imports from util, web, cli folders
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import json
import re
from datetime import UTC, datetime
import math
from typing import Any, Sequence

from util.llm_queries import get_openai_client, log_llm_call
from util.data_store import DataStore
from util.query_dblp import DblpQueryEngine, Publication, get_target_publications_for_author


DEFAULT_MODEL = "text-embedding-3-small"
MIN_YEAR = 2022
MAX_YEAR = 2026
TOPIC_RECORD_PREFIX = "__topic__:"


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_topic_text(topic: str) -> str:
    """Normalize topic text for cache keys and comparisons."""
    return normalize_whitespace(topic).casefold()


def topic_record_name(topic: str) -> str:
    """Derive stable record key for cached topic embeddings."""
    normalized_topic = normalize_topic_text(topic)
    if not normalized_topic:
        raise ValueError("topic must be non-empty")
    return f"{TOPIC_RECORD_PREFIX}{normalized_topic}"


def _build_topic_embedding_text(topic: str) -> str:
    normalized_topic = normalize_topic_text(topic)
    return f"Research topic: {normalized_topic}"


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


def cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float | None:
    """Compute cosine similarity between two vectors."""
    if len(lhs) != len(rhs) or not lhs:
        return None

    lhs_norm_sq = 0.0
    rhs_norm_sq = 0.0
    dot_product = 0.0
    for left_value, right_value in zip(lhs, rhs, strict=True):
        left = float(left_value)
        right = float(right_value)
        dot_product += left * right
        lhs_norm_sq += left * left
        rhs_norm_sq += right * right

    if lhs_norm_sq <= 1e-12 or rhs_norm_sq <= 1e-12:
        return None

    return dot_product / math.sqrt(lhs_norm_sq * rhs_norm_sq)


def get_or_create_topic_embedding(
    store: DataStore,
    topic: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any] | None:
    """Fetch cached topic embedding or create and persist it."""
    normalized_topic = normalize_topic_text(topic)
    if not normalized_topic:
        return None

    record_name = topic_record_name(normalized_topic)
    embeddings_by_name = store.load_expertise_embeddings()
    existing = embeddings_by_name.get(record_name)
    if existing:
        existing_vector = _clean_embedding_vector(existing.get("embedding"))
        if existing_vector is not None:
            return {**existing, "embedding": existing_vector}

    vectors = embed_texts([_build_topic_embedding_text(normalized_topic)], model=model, batch_size=1)
    if not vectors:
        return None

    update = {
        "name": record_name,
        "kind": "topic",
        "topic": normalized_topic,
        "embedding": vectors[0],
        "model": model,
        "updated_at": now_iso(),
    }
    store.update_many_expertise([update])
    return update


def _format_publication(publication: Publication) -> str:
    parts = [
        normalize_whitespace(publication.title or ""),
        normalize_whitespace(publication.venue or ""),
        str(publication.year) if publication.year is not None else "",
    ]
    return " | ".join(part for part in parts if part)


def _build_expertise_text(person_name: str, publications: Sequence[Publication]) -> str:
    lines = [
        f"Name: {person_name}",
        f"Expertise evidence: target-venue publications in years {MIN_YEAR}-{MAX_YEAR}.",
    ]

    if publications:
        lines.append("Recent papers:")
        for publication in sorted(publications, key=lambda pub: (pub.year or 0, pub.title or ""), reverse=True):
            lines.append(f"- {_format_publication(publication)}")
    else:
        lines.append("Recent papers: none in target venues during this time window.")

    return "\n".join(lines)


def embed_texts(texts: Sequence[str], model: str = DEFAULT_MODEL, batch_size: int = 64) -> list[list[float]]:
    if not texts:
        return []

    client = get_openai_client()
    vectors: list[list[float]] = []
    total_input_tokens = 0

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=chunk)
        for row in sorted(response.data, key=lambda item: item.index):
            vectors.append(list(row.embedding))
        if response.usage:
            total_input_tokens += response.usage.prompt_tokens

    sample = texts[0][:300] + "..." if len(texts[0]) > 300 else texts[0]
    prompt_summary = f"{len(texts)} text(s) to embed (model={model}). First text:\n{sample}"
    dim = len(vectors[0]) if vectors else 0
    response_summary = f"{len(vectors)} embedding vector(s) returned, dim={dim}"
    log_llm_call("compute expertise embeddings", prompt_summary, response_summary, total_input_tokens, 0)

    return vectors


def add_expertise_embeddings(
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    limit: int | None = None,
    dry_run: bool = False,
    recompute_all: bool = False,
) -> tuple[int, int]:
    """Add expertise embeddings to all people records.

    By default, skips people who already have embeddings stored.
    Pass recompute_all=True to recompute embeddings for everyone.

    Returns:
        Tuple of (updated_people_count, without_target_publications_count).
    """
    store = DataStore()
    people = store.load()
    years_back = MAX_YEAR - MIN_YEAR
    engine = DblpQueryEngine(preload_index=True)

    names = sorted(people.keys(), key=str.casefold)
    if limit is not None:
        names = names[:limit]

    # Skip people who already have embeddings, unless recomputing all
    if not recompute_all:
        existing_embeddings = store.load_expertise_embeddings()
        names = [name for name in names if name not in existing_embeddings]

    embedding_inputs: list[str] = []
    publication_counts: list[int] = []

    for person_name in names:
        publications = get_target_publications_for_author(
            person_name,
            engine,
            current_year=MAX_YEAR,
            max_years_back=years_back,
        )
        # Keep the final year gate explicit for this script's fixed scope.
        publications = [pub for pub in publications if pub.year is not None and MIN_YEAR <= pub.year <= MAX_YEAR]
        publication_counts.append(len(publications))
        embedding_inputs.append(_build_expertise_text(person_name, publications))

    embeddings = embed_texts(embedding_inputs, model=model, batch_size=batch_size)

    updates: list[dict[str, Any]] = []
    for person_name, publication_count, embedding in zip(names, publication_counts, embeddings, strict=True):
        updates.append(
            {
                "name": person_name,
                "embedding": embedding,
                "model": model,
                "publication_count": publication_count,
                "year_range": [MIN_YEAR, MAX_YEAR],
                "updated_at": now_iso(),
            }
        )

    without_target_publications = sum(1 for count in publication_counts if count == 0)

    if dry_run:
        return len(updates), without_target_publications

    added, updated = store.update_many_expertise(updates)
    return added + updated, without_target_publications


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add an expertise embedding to each person, based on target-venue "
            f"publications in years {MIN_YEAR}-{MAX_YEAR}"
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N people (sorted by name)")
    parser.add_argument("--dry-run", action="store_true", help="Compute embeddings but do not write updates")
    parser.add_argument("--recompute_all", action="store_true", help="Recompute embeddings for people who already have them")
    parser.add_argument(
        "--topic",
        action="append",
        default=None,
        help="Optional topic to precompute and cache in expertise_embeddings.jsonl (can be repeated).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.topic:
        store = DataStore()
        created_topics: list[str] = []
        for raw_topic in args.topic:
            if not isinstance(raw_topic, str) or not raw_topic.strip():
                continue
            topic_record = get_or_create_topic_embedding(store, raw_topic, model=args.model)
            if topic_record is not None:
                created_topics.append(topic_record.get("topic", ""))

        print(
            json.dumps(
                {
                    "cached_topics": created_topics,
                    "model": args.model,
                },
                indent=2,
            )
        )
        return 0

    updated_count, without_target_publications = add_expertise_embeddings(
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        recompute_all=args.recompute_all,
    )

    print(
        json.dumps(
            {
                "updated_people": updated_count,
                "years": [MIN_YEAR, MAX_YEAR],
                "without_target_publications": without_target_publications,
                "dry_run": args.dry_run,
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
