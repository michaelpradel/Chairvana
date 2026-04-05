from __future__ import annotations

import math
from typing import Any


def clean_embedding_vector(raw_embedding: Any) -> list[float] | None:
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


def dot(lhs: list[float], rhs: list[float]) -> float:
    return sum(a * b for a, b in zip(lhs, rhs, strict=True))


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def normalize_vector(vector: list[float]) -> list[float] | None:
    norm = vector_norm(vector)
    if norm <= 1e-12:
        return None
    return [value / norm for value in vector]


def matvec_cov(centered_vectors: list[list[float]], vector: list[float]) -> list[float]:
    sample_count = len(centered_vectors)
    if sample_count <= 1:
        return [0.0 for _ in vector]

    scalar_projections = [dot(row, vector) for row in centered_vectors]
    result = [0.0 for _ in vector]
    for row, projection in zip(centered_vectors, scalar_projections, strict=True):
        for index, value in enumerate(row):
            result[index] += projection * value

    scale = 1.0 / (sample_count - 1)
    return [value * scale for value in result]


def principal_component(
    centered_vectors: list[list[float]],
    basis_vectors: list[list[float]],
    max_iterations: int = 12,
) -> list[float] | None:
    if not centered_vectors:
        return None

    dimension = len(centered_vectors[0])
    candidate = centered_vectors[0][:]
    if vector_norm(candidate) <= 1e-12:
        candidate = [1.0 for _ in range(dimension)]

    for base in basis_vectors:
        projection = dot(candidate, base)
        candidate = [value - projection * base_value for value, base_value in zip(candidate, base, strict=True)]

    normalized_candidate = normalize_vector(candidate)
    if normalized_candidate is None:
        return None

    candidate = normalized_candidate
    for _ in range(max_iterations):
        next_candidate = matvec_cov(centered_vectors, candidate)
        for base in basis_vectors:
            projection = dot(next_candidate, base)
            next_candidate = [
                value - projection * base_value
                for value, base_value in zip(next_candidate, base, strict=True)
            ]

        normalized_next = normalize_vector(next_candidate)
        if normalized_next is None:
            return None

        delta = vector_norm([
            a - b for a, b in zip(normalized_next, candidate, strict=True)
        ])
        candidate = normalized_next
        if delta <= 1e-6:
            break

    return candidate


def project_embeddings_2d(embeddings: list[list[float]]) -> list[tuple[float, float]]:
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

    first_component = principal_component(centered, basis_vectors=[])
    if first_component is None:
        return [(0.0, 0.0) for _ in embeddings]

    second_component = principal_component(centered, basis_vectors=[first_component])
    if second_component is None:
        second_component = [0.0 for _ in range(dimension)]
        if dimension > 1:
            second_component[1] = 1.0

    xs = [dot(vector, first_component) for vector in centered]
    ys = [dot(vector, second_component) for vector in centered]

    x_scale = max((abs(value) for value in xs), default=1.0)
    y_scale = max((abs(value) for value in ys), default=1.0)
    x_scale = x_scale if x_scale > 1e-9 else 1.0
    y_scale = y_scale if y_scale > 1e-9 else 1.0

    return [(x / x_scale, y / y_scale) for x, y in zip(xs, ys, strict=True)]
