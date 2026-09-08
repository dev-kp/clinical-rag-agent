from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedResult:
    item_id: str
    score: float


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = DEFAULT_RRF_K) -> list[FusedResult]:
    """Combine multiple ranked lists of the same kind of item (e.g. one ranking
    from vector similarity search, one from full-text keyword search) into a
    single fused ranking, using Reciprocal Rank Fusion.

    Each list in `rankings` is already ordered best-to-worst, containing item
    ids (e.g. chunk_id). An item does not need to appear in every ranking.

    Returns FusedResult objects sorted by fused score, descending.
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)

    return sorted(
        (FusedResult(item_id, score) for item_id, score in scores.items()),
        key=lambda r: r.score,
        reverse=True,
    )
