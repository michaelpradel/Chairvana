"""Create and query person embeddings for expertise discovery.

This module uses OpenAI's ``text-embedding-3-small`` model to:
1. Build embeddings for people from papers and research topics.
2. Query the embedding space to find experts for a term.
3. Compute nearest neighbors and k-means clusters.

Embeddings are persisted to ``data/person_embeddings.json`` and can be
incrementally updated with additional people later.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from llm_queries import get_openai_client


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "person_embeddings.json"


def now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    lowered = normalize_whitespace(value).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "person"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def average_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average an empty list of vectors")
    size = len(vectors[0])
    sums = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            raise ValueError("All vectors must have the same dimensionality")
        for index, value in enumerate(vector):
            sums[index] += value
    count = float(len(vectors))
    return [value / count for value in sums]


@dataclass(slots=True)
class PersonRecord:
    person_id: str
    name: str
    topics: list[str]
    papers: list[dict[str, Any]]
    source_text: str


@dataclass(slots=True)
class RankedPerson:
    person_id: str
    name: str
    score: float


@dataclass(slots=True)
class Cluster:
    cluster_id: int
    members: list[str]
    names: list[str]


def build_person_text(person: dict[str, Any]) -> str:
    name = normalize_whitespace(str(person.get("name", "")))
    if not name:
        raise ValueError("Each person must contain a non-empty 'name'")

    topics_raw = person.get("topics", [])
    topics = [normalize_whitespace(str(topic)) for topic in topics_raw if str(topic).strip()]

    papers_raw = person.get("papers", [])
    paper_lines: list[str] = []
    for paper in papers_raw:
        if isinstance(paper, str):
            line = normalize_whitespace(paper)
            if line:
                paper_lines.append(line)
            continue

        if not isinstance(paper, dict):
            continue

        title = normalize_whitespace(str(paper.get("title", "")))
        venue = normalize_whitespace(str(paper.get("venue", "")))
        year = str(paper.get("year", "")).strip()
        abstract = normalize_whitespace(str(paper.get("abstract", "")))

        parts = [part for part in (title, venue, year) if part]
        line = " | ".join(parts)
        if abstract:
            line = f"{line}. Abstract: {abstract}" if line else f"Abstract: {abstract}"
        if line:
            paper_lines.append(line)

    lines = [f"Name: {name}"]
    if topics:
        lines.append("Research topics: " + "; ".join(topics))
    if paper_lines:
        lines.append("Papers:")
        lines.extend(f"- {line}" for line in paper_lines)

    return "\n".join(lines)


def parse_people(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing people input file: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        people: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                content = line.strip()
                if not content:
                    continue
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("Each JSONL line must be an object")
                people.append(parsed)
        return people

    parsed = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict) and isinstance(parsed.get("people"), list):
        items = parsed["people"]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("Input must be a list of person objects or {'people': [...]}.")

    if not all(isinstance(item, dict) for item in items):
        raise ValueError("All people entries must be JSON objects")

    return [item for item in items if isinstance(item, dict)]


def person_to_record(person: dict[str, Any], existing_ids: set[str]) -> PersonRecord:
    name = normalize_whitespace(str(person.get("name", "")))
    if not name:
        raise ValueError("Each person must contain a non-empty 'name'")

    requested_id = normalize_whitespace(str(person.get("id", "")))
    person_id = requested_id or slugify(name)

    # Ensure deterministic uniqueness for generated IDs.
    if person_id in existing_ids and not requested_id:
        suffix = 2
        while f"{person_id}-{suffix}" in existing_ids:
            suffix += 1
        person_id = f"{person_id}-{suffix}"

    topics_raw = person.get("topics", [])
    topics = [normalize_whitespace(str(topic)) for topic in topics_raw if str(topic).strip()]

    papers_raw = person.get("papers", [])
    papers: list[dict[str, Any]] = []
    for paper in papers_raw:
        if isinstance(paper, dict):
            papers.append(paper)
        elif isinstance(paper, str):
            papers.append({"title": normalize_whitespace(paper)})

    source_text = build_person_text(person)
    return PersonRecord(person_id=person_id, name=name, topics=topics, papers=papers, source_text=source_text)


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


def load_store(store_path: Path) -> dict[str, Any]:
    if not store_path.exists():
        return {
            "model": DEFAULT_MODEL,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "people": {},
        }

    parsed = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid embedding store format: {store_path}")

    parsed.setdefault("model", DEFAULT_MODEL)
    parsed.setdefault("created_at", now_iso())
    parsed.setdefault("updated_at", now_iso())
    parsed.setdefault("people", {})

    if not isinstance(parsed["people"], dict):
        raise ValueError(f"Invalid 'people' map in embedding store: {store_path}")

    return parsed


def save_store(store: dict[str, Any], store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = now_iso()
    store_path.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_or_update_people(
    people: Sequence[dict[str, Any]],
    store_path: Path = DEFAULT_STORE_PATH,
    model: str = DEFAULT_MODEL,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    store = load_store(store_path)
    store["model"] = model

    people_map: dict[str, dict[str, Any]] = store["people"]
    existing_ids = set(people_map.keys())

    records: list[PersonRecord] = []
    for person in people:
        record = person_to_record(person, existing_ids)
        if record.person_id in people_map and not overwrite_existing:
            continue
        existing_ids.add(record.person_id)
        records.append(record)

    embeddings = embed_texts([record.source_text for record in records], model=model)

    for record, vector in zip(records, embeddings, strict=True):
        people_map[record.person_id] = {
            "name": record.name,
            "topics": record.topics,
            "papers": record.papers,
            "source_text": record.source_text,
            "embedding": vector,
            "updated_at": now_iso(),
        }

    save_store(store, store_path)
    return store


def query_experts(term: str, store_path: Path = DEFAULT_STORE_PATH, top_k: int = 10) -> list[RankedPerson]:
    store = load_store(store_path)
    people_map: dict[str, dict[str, Any]] = store["people"]
    if not people_map:
        return []

    [query_vector] = embed_texts([term], model=str(store["model"]))

    ranked: list[RankedPerson] = []
    for person_id, payload in people_map.items():
        vector = payload.get("embedding")
        if not isinstance(vector, list):
            continue
        score = cosine_similarity(query_vector, [float(value) for value in vector])
        ranked.append(RankedPerson(person_id=person_id, name=str(payload.get("name", person_id)), score=score))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def nearest_neighbors(
    person_id: str,
    store_path: Path = DEFAULT_STORE_PATH,
    top_k: int = 5,
) -> list[RankedPerson]:
    store = load_store(store_path)
    people_map: dict[str, dict[str, Any]] = store["people"]
    if person_id not in people_map:
        raise KeyError(f"Unknown person id: {person_id}")

    anchor = people_map[person_id].get("embedding")
    if not isinstance(anchor, list):
        raise ValueError(f"Person has no embedding: {person_id}")

    anchor_vector = [float(value) for value in anchor]

    ranked: list[RankedPerson] = []
    for candidate_id, payload in people_map.items():
        if candidate_id == person_id:
            continue
        vector = payload.get("embedding")
        if not isinstance(vector, list):
            continue
        score = cosine_similarity(anchor_vector, [float(value) for value in vector])
        ranked.append(
            RankedPerson(
                person_id=candidate_id,
                name=str(payload.get("name", candidate_id)),
                score=score,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def cluster_people(
    k: int,
    store_path: Path = DEFAULT_STORE_PATH,
    seed: int = 0,
    max_iters: int = 50,
) -> list[Cluster]:
    if k <= 0:
        raise ValueError("k must be positive")

    store = load_store(store_path)
    people_map: dict[str, dict[str, Any]] = store["people"]

    entries: list[tuple[str, str, list[float]]] = []
    for person_id, payload in people_map.items():
        vector = payload.get("embedding")
        if isinstance(vector, list) and vector:
            entries.append((person_id, str(payload.get("name", person_id)), normalize_vector(vector)))

    if not entries:
        raise ValueError("No embeddings available for clustering")
    if k > len(entries):
        raise ValueError(f"k ({k}) cannot exceed number of people ({len(entries)})")

    rng = random.Random(seed)
    centroids = [list(entry[2]) for entry in rng.sample(entries, k)]

    assignments = [-1] * len(entries)
    for _ in range(max_iters):
        changed = False

        for idx, (_, _, vector) in enumerate(entries):
            best_cluster = 0
            best_score = -1.0
            for cluster_id, centroid in enumerate(centroids):
                score = cosine_similarity(vector, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id

            if assignments[idx] != best_cluster:
                assignments[idx] = best_cluster
                changed = True

        grouped: list[list[list[float]]] = [[] for _ in range(k)]
        for idx, assignment in enumerate(assignments):
            grouped[assignment].append(entries[idx][2])

        for cluster_id in range(k):
            if grouped[cluster_id]:
                centroids[cluster_id] = normalize_vector(average_vectors(grouped[cluster_id]))
            else:
                centroids[cluster_id] = list(entries[rng.randrange(len(entries))][2])

        if not changed:
            break

    output: list[Cluster] = []
    for cluster_id in range(k):
        member_ids: list[str] = []
        member_names: list[str] = []
        for idx, assignment in enumerate(assignments):
            if assignment == cluster_id:
                person_key, person_name, _ = entries[idx]
                member_ids.append(person_key)
                member_names.append(person_name)
        output.append(Cluster(cluster_id=cluster_id, members=member_ids, names=member_names))

    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query person embeddings")
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help="Embedding store path (default: data/person_embeddings.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Create/update embeddings from people JSON/JSONL")
    embed_parser.add_argument("--input", type=Path, required=True, help="Path to people JSON or JSONL file")
    embed_parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    embed_parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute embeddings for people whose IDs already exist in the store",
    )

    query_parser = subparsers.add_parser("query", help="Find experts for a textual query")
    query_parser.add_argument("--term", required=True, help="Query term, e.g., 'program analysis'")
    query_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")

    neighbors_parser = subparsers.add_parser("neighbors", help="Find nearest neighbors for a person")
    neighbors_parser.add_argument("--person-id", required=True, help="Person ID in embedding store")
    neighbors_parser.add_argument("--top-k", type=int, default=5, help="Number of neighbors to return")

    cluster_parser = subparsers.add_parser("cluster", help="Cluster people in embedding space")
    cluster_parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    cluster_parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    cluster_parser.add_argument("--max-iters", type=int, default=50, help="Maximum k-means iterations")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "embed":
        people = parse_people(args.input)
        store = add_or_update_people(
            people,
            store_path=args.store,
            model=args.model,
            overwrite_existing=args.overwrite_existing,
        )
        count = len(store.get("people", {}))
        print(json.dumps({"store": str(args.store), "people": count, "model": store.get("model")}, indent=2))
        return 0

    if args.command == "query":
        results = query_experts(args.term, store_path=args.store, top_k=args.top_k)
        print(
            json.dumps(
                [
                    {"person_id": item.person_id, "name": item.name, "score": round(item.score, 6)}
                    for item in results
                ],
                indent=2,
            )
        )
        return 0

    if args.command == "neighbors":
        results = nearest_neighbors(args.person_id, store_path=args.store, top_k=args.top_k)
        print(
            json.dumps(
                [
                    {"person_id": item.person_id, "name": item.name, "score": round(item.score, 6)}
                    for item in results
                ],
                indent=2,
            )
        )
        return 0

    if args.command == "cluster":
        clusters = cluster_people(
            args.k,
            store_path=args.store,
            seed=args.seed,
            max_iters=args.max_iters,
        )
        print(
            json.dumps(
                [
                    {
                        "cluster_id": cluster.cluster_id,
                        "size": len(cluster.members),
                        "members": cluster.members,
                        "names": cluster.names,
                    }
                    for cluster in clusters
                ],
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc
