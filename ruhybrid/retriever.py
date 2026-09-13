"""Hybrid retrieval: fuse BM25 and dense rankings with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ruhybrid.chunker import Chunk

if TYPE_CHECKING:
    from ruhybrid.index import HybridIndex

RRF_K = 60


@dataclass
class SearchHit:
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    chunk: Chunk


@dataclass
class FusedItem:
    item: int
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    rank_lists: dict[str, list[int]], k: int = RRF_K
) -> list[FusedItem]:
    """Fuse ranked id lists by summing 1 / (k + rank), rank starting at 1.

    RRF needs only ranks, so BM25 scores and cosine similarities never have to
    share a scale. See Cormack, Clarke and Buettcher 2009.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for method, ranking in rank_lists.items():
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(item, {})[method] = rank
    fused = [FusedItem(item=i, score=s, ranks=ranks[i]) for i, s in scores.items()]
    fused.sort(key=lambda f: f.score, reverse=True)
    return fused


def _candidate_count(top_k: int) -> int:
    # pull more candidates than requested from each method so fusion can rescue
    # a document one side ranks deep; the floor of 30 matters for small top_k
    return max(30, min(200, top_k * 4))


class HybridRetriever:
    """Fuse the index's sparse and dense rankings with RRF.

    Unweighted RRF uses ranks only, so it can drag the stronger retriever down
    toward the weaker one. Measure bm25 and dense separately first; if one is
    roughly 1.5-2x the other, prefer weighted fusion or a reranker.
    """

    def __init__(self, index: HybridIndex, k: int = RRF_K):
        self.index = index
        self.k = k

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query or not query.strip():
            return []
        n = _candidate_count(top_k)
        sparse = self.index.search_sparse(query, n)
        dense = self.index.search_dense(query, n)
        fused = reciprocal_rank_fusion({"bm25": sparse, "dense": dense}, self.k)

        # keep only the best chunk of each document; without this a single long
        # article can take several of the top slots with its overlapping chunks
        hits = []
        seen_docs = set()
        for f in fused:
            chunk = self.index.chunks[f.item]
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            hits.append(
                SearchHit(
                    score=f.score,
                    bm25_rank=f.ranks.get("bm25"),
                    dense_rank=f.ranks.get("dense"),
                    chunk=chunk,
                )
            )
            if len(hits) >= top_k:
                break
        return hits
