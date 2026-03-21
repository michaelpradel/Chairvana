"""Compute and store expertise embeddings for people.

This script computes one embedding per person from recent DBLP publications,
using the same target-venue filtering used by sync_people_with_publications.
Embeddings are written to expertise_embeddings.jsonl, while people.jsonl remains
human-readable.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from typing import Any, Sequence

from llm_queries import get_openai_client
from people import PeopleStore
from query_dblp import DblpQueryEngine, Publication, get_target_publications_for_author


DEFAULT_MODEL = "text-embedding-3-small"
MIN_YEAR = 2022
MAX_YEAR = 2026


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=chunk)
        for row in sorted(response.data, key=lambda item: item.index):
            vectors.append(list(row.embedding))

    return vectors


def add_expertise_embeddings(
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Add expertise embeddings to all people records.

    Returns:
        Tuple of (updated_people_count, without_target_publications_count).
    """
    store = PeopleStore()
    people = store.load()
    years_back = MAX_YEAR - MIN_YEAR
    engine = DblpQueryEngine(preload_index=True)

    names = sorted(people.keys(), key=str.casefold)
    if limit is not None:
        names = names[:limit]

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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    updated_count, without_target_publications = add_expertise_embeddings(
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
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
