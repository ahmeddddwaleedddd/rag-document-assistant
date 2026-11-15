"""Phase 6 metrics as pure functions: no I/O, no API, unit-tested exactly.

Relevance is judged by PAGE, not chunk id, on purpose: chunk ids change whenever
CHUNK_SIZE changes, page numbers don't, so one golden set survives every tuning run.
"""

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved_pages: Sequence[int], expected_pages: Iterable[int], k: int) -> float:
    """1.0 if ANY expected page appears in the top-k, else 0.0 (also called hit@k).
    Averaged over questions, this gives the 'recall@k' number quoted in the README."""
    expected = set(expected_pages)
    if not expected:
        raise ValueError("recall_at_k needs at least one expected page")
    return 1.0 if expected & set(retrieved_pages[:k]) else 0.0


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Iterable[int]) -> float:
    """1/rank of the first relevant result; 0.0 if none was retrieved."""
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return 1.0 / rank
    return 0.0


def overlap(a: Iterable, b: Iterable) -> float:
    """Top-k agreement: |A ∩ B| / |B|. Used for 'ANN store vs brute-force numpy'."""
    a, b = set(a), set(b)
    return 1.0 if not b else len(a & b) / len(b)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). No numpy needed, and exact for small n."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def refusal_accuracy(pairs: Sequence[tuple[bool, bool]]) -> float:
    """pairs = [(expected_answerable, actual_answerable), ...]. Share that match."""
    return mean([1.0 if exp == act else 0.0 for exp, act in pairs])
