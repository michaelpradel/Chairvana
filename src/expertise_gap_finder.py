"""Find expertise coverage gaps for tagged people against recent target-venue papers.

This tool compares tagged people's expertise embeddings with embeddings of paper titles
from target venues in a year range, and reports papers that have no nearby tagged
person.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

from add_expertise_embeddings import DEFAULT_MODEL, cosine_similarity, embed_texts
from data_store import DataStore
from query_dblp import (
    DblpQueryEngine,
    Publication,
    TARGET_VENUE_PREFIXES,
    build_publication,
    is_main_track_publication,
)


DEFAULT_MIN_YEAR = 2024
DEFAULT_MAX_YEAR = 2026
DEFAULT_MIN_SIMILARITY = 0.30
DEFAULT_TOP_K = 5
PAPER_MATRIX_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "expertise_gap_paper_matrix_index.npz"


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _normalize_tag(raw_tag: str) -> str:
    normalized = raw_tag.strip().casefold()
    if not normalized:
        raise ValueError("tag must be non-empty")
    if not normalized.startswith("#"):
        normalized = f"#{normalized}"
    if normalized == "#":
        raise ValueError("tag must be non-empty")
    return normalized


def _normalize_flags(raw_flags: Any) -> set[str]:
    if not isinstance(raw_flags, list):
        return set()

    normalized: set[str] = set()
    for flag in raw_flags:
        if not isinstance(flag, str):
            continue
        token = flag.strip().casefold()
        if not token:
            continue
        if not token.startswith("#"):
            token = f"#{token}"
        if token != "#":
            normalized.add(token)
    return normalized


def _paper_id(publication: Publication) -> str:
    if isinstance(publication.key, str) and publication.key.strip():
        return publication.key.strip()

    title = publication.title or ""
    venue = publication.venue or ""
    year = publication.year if isinstance(publication.year, int) else 0
    digest = hashlib.sha1(f"{title}|{venue}|{year}".encode("utf-8")).hexdigest()[:16]
    return f"fallback:{digest}"


def _paper_record_name(publication: Publication) -> str:
    return f"paper:{_paper_id(publication)}"


def _paper_text(publication: Publication) -> str:
    title = publication.title or "Untitled"
    venue = publication.venue or "Unknown venue"
    year = publication.year if isinstance(publication.year, int) else "Unknown year"
    return f"Paper title: {title}\nVenue: {venue}\nYear: {year}"


def _clean_embedding(raw_embedding: Any) -> list[float] | None:
    if not isinstance(raw_embedding, list) or len(raw_embedding) < 2:
        return None

    vector: list[float] = []
    for value in raw_embedding:
        if not isinstance(value, (int, float)):
            return None
        vector.append(float(value))
    return vector


def _paper_signature(
    paper_records: list[tuple[str, dict[str, Any]]],
    *,
    min_year: int,
    max_year: int,
) -> str:
    digest = hashlib.sha1()
    digest.update(f"years:{min_year}-{max_year}\n".encode("utf-8"))
    for record_name, record in paper_records:
        embedding = _clean_embedding(record.get("embedding"))
        if embedding is None:
            continue

        digest.update(record_name.encode("utf-8"))
        digest.update(b"|")
        digest.update(str(record.get("updated_at") or "").encode("utf-8"))
        digest.update(b"|")
        digest.update(str(record.get("model") or "").encode("utf-8"))
        digest.update(b"|")
        digest.update(str(len(embedding)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_vector(vector: list[float]) -> list[float] | None:
    norm_sq = sum(value * value for value in vector)
    if norm_sq <= 1e-12:
        return None
    norm = norm_sq ** 0.5
    return [value / norm for value in vector]


def _build_paper_matrix_index(
    *,
    papers: list[Publication],
    paper_embeddings: dict[str, dict[str, Any]],
    min_year: int,
    max_year: int,
) -> dict[str, Any] | None:
    if np is None:
        return None

    rows: list[list[float]] = []
    record_names: list[str] = []
    paper_records: list[tuple[str, dict[str, Any]]] = []

    for paper in papers:
        record_name = _paper_record_name(paper)
        record = paper_embeddings.get(record_name)
        if not isinstance(record, dict):
            continue

        vector = _clean_embedding(record.get("embedding"))
        if vector is None:
            continue

        normalized = _normalize_vector(vector)
        if normalized is None:
            continue

        rows.append(normalized)
        record_names.append(record_name)
        paper_records.append((record_name, record))

    if not rows:
        return None

    matrix = np.asarray(rows, dtype=np.float32)
    signature = _paper_signature(paper_records, min_year=min_year, max_year=max_year)
    return {
        "version": 2,
        "created_at": now_iso(),
        "min_year": min_year,
        "max_year": max_year,
        "paper_signature": signature,
        "record_names": record_names,
        "matrix": matrix,
    }


def _save_paper_matrix_index(path: Path, payload: dict[str, Any]) -> None:
    if np is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": payload.get("version", 2),
        "created_at": payload.get("created_at", now_iso()),
        "min_year": payload.get("min_year"),
        "max_year": payload.get("max_year"),
        "paper_signature": payload.get("paper_signature"),
    }
    np.savez_compressed(
        path,
        meta=np.asarray([json.dumps(meta, ensure_ascii=False)], dtype=np.str_),
        record_names=np.asarray(payload.get("record_names", []), dtype=np.str_),
        matrix=np.asarray(payload.get("matrix"), dtype=np.float32),
    )


def _load_paper_matrix_index(path: Path) -> dict[str, Any] | None:
    if np is None or not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as loaded:
            if "meta" not in loaded or "record_names" not in loaded or "matrix" not in loaded:
                return None

            raw_meta = loaded["meta"]
            if raw_meta.size == 0:
                return None
            meta = json.loads(str(raw_meta[0]))
            matrix = np.asarray(loaded["matrix"], dtype=np.float32)
            record_names = [str(value) for value in loaded["record_names"].tolist()]
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(meta, dict):
        return None
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] != len(record_names):
        return None

    return {
        "version": meta.get("version", 2),
        "created_at": meta.get("created_at"),
        "min_year": meta.get("min_year"),
        "max_year": meta.get("max_year"),
        "paper_signature": meta.get("paper_signature"),
        "record_names": record_names,
        "matrix": matrix,
    }


def _exact_closest_people(
    paper_vector: list[float],
    person_vectors: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    closest: list[dict[str, Any]] = []
    for person in person_vectors:
        similarity = cosine_similarity(person["embedding"], paper_vector)
        if similarity is None:
            continue
        closest.append(
            {
                "name": person["name"],
                "affiliation": person.get("affiliation", ""),
                "country": person.get("country", ""),
                "similarity": round(similarity, 4),
            }
        )

    closest.sort(key=lambda row: (-row["similarity"], row["name"].casefold()))
    return closest[:top_k]


def _collect_target_papers(min_year: int, max_year: int, engine: DblpQueryEngine) -> list[Publication]:
    papers_by_id: dict[str, Publication] = {}

    with engine.filtered_jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            row = line.strip()
            if not row:
                continue

            publication = build_publication(json.loads(row))
            if not isinstance(publication.year, int) or publication.year < min_year or publication.year > max_year:
                continue

            if not any(is_main_track_publication(publication, prefix, engine) for prefix in TARGET_VENUE_PREFIXES):
                continue

            paper_id = _paper_id(publication)
            papers_by_id.setdefault(paper_id, publication)

    return sorted(
        papers_by_id.values(),
        key=lambda pub: (
            -(pub.year if isinstance(pub.year, int) else -1),
            (pub.title or "").casefold(),
        ),
    )


def _ensure_paper_embeddings(
    papers: list[Publication],
    *,
    store: DataStore,
    model: str,
    batch_size: int,
    recompute: bool,
    persist_updates: bool,
    base_commit: str | None,
) -> dict[str, dict[str, Any]]:
    existing = store.load_paper_expertise_embeddings(commit=base_commit)

    to_embed: list[Publication] = []
    for paper in papers:
        record_name = _paper_record_name(paper)
        if recompute:
            to_embed.append(paper)
            continue

        existing_record = existing.get(record_name)
        if not isinstance(existing_record, dict):
            to_embed.append(paper)
            continue

        vector = _clean_embedding(existing_record.get("embedding"))
        if vector is None:
            to_embed.append(paper)

    if not to_embed:
        return existing

    vectors = embed_texts([_paper_text(paper) for paper in to_embed], model=model, batch_size=batch_size)

    updates: list[dict[str, Any]] = []
    for paper, vector in zip(to_embed, vectors, strict=True):
        updates.append(
            {
                "name": _paper_record_name(paper),
                "paper_id": _paper_id(paper),
                "dblp_key": paper.key,
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.year,
                "model": model,
                "updated_at": now_iso(),
                "embedding": vector,
            }
        )

    if persist_updates:
        store.update_many_paper_expertise(updates, base_commit=base_commit)
        return store.load_paper_expertise_embeddings(commit=base_commit)

    merged = dict(existing)
    for update in updates:
        update_name = update.get("name")
        if isinstance(update_name, str) and update_name:
            merged[update_name] = update
    return merged


def find_expertise_gaps(
    *,
    tag: str,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    top_k: int = DEFAULT_TOP_K,
    max_results: int | None = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    recompute_paper_embeddings: bool = False,
    recompute_similarity_index: bool = False,
    base_commit: str | None = None,
) -> dict[str, Any]:
    if min_year > max_year:
        raise ValueError("min_year must be <= max_year")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    normalized_tag = _normalize_tag(tag)
    store = DataStore()

    people = store.list_people(commit=base_commit)
    tagged_people = [
        person
        for person in people
        if normalized_tag in _normalize_flags(person.get("flags"))
        and isinstance(person.get("name"), str)
        and person.get("name", "").strip()
    ]

    person_embeddings = store.load_expertise_embeddings(commit=base_commit)
    tagged_vectors: list[dict[str, Any]] = []
    tagged_without_embeddings: list[str] = []

    for person in tagged_people:
        name = str(person["name"]).strip()
        embedding_record = person_embeddings.get(name)
        vector = _clean_embedding(embedding_record.get("embedding") if isinstance(embedding_record, dict) else None)
        if vector is None:
            tagged_without_embeddings.append(name)
            continue

        tagged_vectors.append(
            {
                "name": name,
                "affiliation": person.get("affiliation") if isinstance(person.get("affiliation"), str) else "",
                "country": person.get("country") if isinstance(person.get("country"), str) else "",
                "embedding": vector,
            }
        )

    engine = DblpQueryEngine(preload_index=True)
    papers = _collect_target_papers(min_year, max_year, engine)
    paper_embeddings = _ensure_paper_embeddings(
        papers,
        store=store,
        model=model,
        batch_size=batch_size,
        recompute=recompute_paper_embeddings,
        persist_updates=base_commit is None,
        base_commit=base_commit,
    )

    paper_records: list[tuple[str, dict[str, Any]]] = []
    for paper in papers:
        record_name = _paper_record_name(paper)
        record = paper_embeddings.get(record_name)
        if not isinstance(record, dict):
            continue
        if _clean_embedding(record.get("embedding")) is None:
            continue
        paper_records.append((record_name, record))

    index_payload: dict[str, Any] | None = None
    used_similarity_index = False
    index_similarities: dict[str, list[float]] = {}
    tagged_vectors_by_index: list[dict[str, Any]] = []

    if base_commit is None and np is not None and paper_records and tagged_vectors:
        target_signature = _paper_signature(paper_records, min_year=min_year, max_year=max_year)
        existing_index = None if recompute_similarity_index else _load_paper_matrix_index(PAPER_MATRIX_INDEX_PATH)

        if (
            existing_index is not None
            and existing_index.get("paper_signature") == target_signature
            and int(existing_index.get("min_year") or -1) == min_year
            and int(existing_index.get("max_year") or -1) == max_year
            and isinstance(existing_index.get("record_names"), list)
            and hasattr(existing_index.get("matrix"), "shape")
        ):
            index_payload = existing_index
            used_similarity_index = True
        else:
            rebuilt_index = _build_paper_matrix_index(
                papers=papers,
                paper_embeddings=paper_embeddings,
                min_year=min_year,
                max_year=max_year,
            )
            if rebuilt_index is not None:
                _save_paper_matrix_index(PAPER_MATRIX_INDEX_PATH, rebuilt_index)
                index_payload = rebuilt_index
                used_similarity_index = True

        if index_payload is not None:
            matrix = index_payload.get("matrix")
            record_names = index_payload.get("record_names")
            if isinstance(record_names, list) and hasattr(matrix, "shape"):
                matrix_2d = np.asarray(matrix, dtype=np.float32)
                tagged_matrix_rows: list[list[float]] = []
                for person in tagged_vectors:
                    normalized = _normalize_vector(person["embedding"])
                    if normalized is None:
                        continue
                    tagged_matrix_rows.append(normalized)
                    tagged_vectors_by_index.append(person)

                if tagged_matrix_rows:
                    tagged_matrix = np.asarray(tagged_matrix_rows, dtype=np.float32)
                    similarities = tagged_matrix @ matrix_2d.T
                    for column_index, record_name in enumerate(record_names):
                        index_similarities[record_name] = similarities[:, column_index].tolist()
                else:
                    used_similarity_index = False

    if not tagged_vectors_by_index:
        tagged_vectors_by_index = tagged_vectors

    gaps: list[dict[str, Any]] = []
    papers_with_embeddings = 0

    for paper in papers:
        record = paper_embeddings.get(_paper_record_name(paper))
        if not isinstance(record, dict):
            continue

        paper_vector = _clean_embedding(record.get("embedding"))
        if paper_vector is None:
            continue

        papers_with_embeddings += 1

        record_name = _paper_record_name(paper)
        closest: list[dict[str, Any]]
        scores_for_paper = index_similarities.get(record_name)
        if scores_for_paper is not None and len(scores_for_paper) == len(tagged_vectors_by_index):
            scored_people: list[dict[str, Any]] = []
            for row_index, similarity in enumerate(scores_for_paper):
                person = tagged_vectors_by_index[row_index]
                scored_people.append(
                    {
                        "name": person["name"],
                        "affiliation": person.get("affiliation", ""),
                        "country": person.get("country", ""),
                        "similarity": round(float(similarity), 4),
                    }
                )
            scored_people.sort(key=lambda row: (-row["similarity"], row["name"].casefold()))
            closest = scored_people[:top_k]
        else:
            closest = _exact_closest_people(paper_vector, tagged_vectors, top_k)

        best_similarity = closest[0]["similarity"] if closest else None
        if best_similarity is None or best_similarity < min_similarity:
            gaps.append(
                {
                    "paper_id": record.get("paper_id") or _paper_id(paper),
                    "dblp_key": paper.key,
                    "title": paper.title or "Untitled",
                    "venue": paper.venue or "Unknown venue",
                    "year": paper.year,
                    "best_similarity": best_similarity,
                    "closest_people": closest,
                }
            )

    gaps.sort(
        key=lambda row: (
            row["best_similarity"] if isinstance(row["best_similarity"], (int, float)) else -1.0,
            -(row["year"] if isinstance(row["year"], int) else -1),
            str(row.get("title") or "").casefold(),
        )
    )

    if max_results is not None and max_results >= 0:
        gaps = gaps[:max_results]

    return {
        "tag": normalized_tag,
        "year_range": [min_year, max_year],
        "min_similarity": min_similarity,
        "top_k": top_k,
        "tagged_people_count": len(tagged_people),
        "tagged_people_with_embeddings": len(tagged_vectors),
        "tagged_people_without_embeddings": tagged_without_embeddings,
        "papers_in_range": len(papers),
        "papers_with_embeddings": papers_with_embeddings,
        "used_similarity_index": used_similarity_index,
        "numpy_available": np is not None,
        "similarity_index_path": str(PAPER_MATRIX_INDEX_PATH),
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def recompute_paper_similarity_index(
    *,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    recompute_paper_embeddings: bool = False,
    base_commit: str | None = None,
) -> dict[str, Any]:
    if min_year > max_year:
        raise ValueError("min_year must be <= max_year")
    if np is None:
        raise RuntimeError("numpy is required to build the paper similarity index")

    store = DataStore()
    engine = DblpQueryEngine(preload_index=True)
    papers = _collect_target_papers(min_year, max_year, engine)
    paper_embeddings = _ensure_paper_embeddings(
        papers,
        store=store,
        model=model,
        batch_size=batch_size,
        recompute=recompute_paper_embeddings,
        persist_updates=base_commit is None,
        base_commit=base_commit,
    )

    rebuilt_index = _build_paper_matrix_index(
        papers=papers,
        paper_embeddings=paper_embeddings,
        min_year=min_year,
        max_year=max_year,
    )
    if rebuilt_index is None:
        return {
            "papers_in_range": len(papers),
            "papers_with_embeddings": 0,
            "index_written": False,
            "similarity_index_path": str(PAPER_MATRIX_INDEX_PATH),
            "numpy_available": np is not None,
        }

    _save_paper_matrix_index(PAPER_MATRIX_INDEX_PATH, rebuilt_index)
    return {
        "papers_in_range": len(papers),
        "papers_with_embeddings": len(rebuilt_index.get("record_names", [])),
        "index_written": True,
        "similarity_index_path": str(PAPER_MATRIX_INDEX_PATH),
        "numpy_available": np is not None,
        "year_range": [min_year, max_year],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find target-venue papers whose title embedding has no nearby tagged person expertise embedding"
        )
    )
    parser.add_argument("--tag", default=None, help="Tag to filter people, e.g., #invite")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help=f"Start year (default: {DEFAULT_MIN_YEAR})")
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR, help=f"End year (default: {DEFAULT_MAX_YEAR})")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"Minimum similarity to count as covered (default: {DEFAULT_MIN_SIMILARITY})",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Closest people per paper (default: {DEFAULT_TOP_K})")
    parser.add_argument("--max-results", type=int, default=200, help="Maximum number of gap papers to print")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--recompute-paper-embeddings", action="store_true", help="Recompute all paper embeddings in range")
    parser.add_argument("--recompute-similarity-index", action="store_true", help="Force rebuilding the paper-only matrix index")
    parser.add_argument("--history", default=None, help="Optional historical commit of people store to inspect")
    args = parser.parse_args(argv)
    if args.tag is None and not args.recompute_similarity_index:
        parser.error("--tag is required unless --recompute-similarity-index is set")
    return args


def _format_similarity(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


if __name__ == "__main__":
    args = parse_args()

    if args.recompute_similarity_index and args.tag is None:
        summary = recompute_paper_similarity_index(
            min_year=args.min_year,
            max_year=args.max_year,
            model=args.model,
            batch_size=args.batch_size,
            recompute_paper_embeddings=args.recompute_paper_embeddings,
            base_commit=args.history,
        )
        print(
            f"Recomputed paper similarity index: {'yes' if summary.get('index_written') else 'no'} | "
            f"papers in range={summary.get('papers_in_range')} | "
            f"papers with embeddings={summary.get('papers_with_embeddings')}"
        )
        print(f"Index path: {summary.get('similarity_index_path')}")
        raise SystemExit(0)

    result = find_expertise_gaps(
        tag=args.tag or "#invite",
        min_year=args.min_year,
        max_year=args.max_year,
        min_similarity=args.min_similarity,
        top_k=args.top_k,
        max_results=args.max_results,
        model=args.model,
        batch_size=args.batch_size,
        recompute_paper_embeddings=args.recompute_paper_embeddings,
        recompute_similarity_index=args.recompute_similarity_index,
        base_commit=args.history,
    )

    print(
        f"Tag {result['tag']} | years {result['year_range'][0]}-{result['year_range'][1]} | "
        f"tagged people: {result['tagged_people_count']} "
        f"({result['tagged_people_with_embeddings']} with embeddings)"
    )
    print(
        f"Target papers: {result['papers_in_range']} ({result['papers_with_embeddings']} with embeddings) | "
        f"gaps: {result['gap_count']}"
    )
    print(
        f"Similarity index: {'used' if result.get('used_similarity_index') else 'not used'}"
        f" ({result.get('similarity_index_path')})"
    )

    if result["tagged_people_without_embeddings"]:
        print("Tagged people without expertise embeddings:")
        for name in result["tagged_people_without_embeddings"][:20]:
            print(f"  - {name}")

    for index, gap in enumerate(result["gaps"], start=1):
        print(
            f"{index}. [{gap.get('year')}] {gap.get('title')}"
            f" ({gap.get('venue')}) best={_format_similarity(gap.get('best_similarity'))}"
        )
        for match in gap.get("closest_people", []):
            print(
                f"    -> {match.get('name')} sim={_format_similarity(match.get('similarity'))}"
            )
