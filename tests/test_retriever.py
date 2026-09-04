from ruhybrid.chunker import Chunk
from ruhybrid.retriever import HybridRetriever, reciprocal_rank_fusion


class FakeIndex:
    """Stand-in for HybridIndex so retriever logic is testable without models."""

    def __init__(self, chunks, sparse, dense):
        self.chunks = chunks
        self._sparse = sparse
        self._dense = dense

    def search_sparse(self, query, n):
        return self._sparse[:n]

    def search_dense(self, query, n):
        return self._dense[:n]


def test_rrf_known_ranks_give_known_score():
    fused = reciprocal_rank_fusion({"bm25": [10, 20], "dense": [20, 10]}, k=60)
    scores = {f.item: f.score for f in fused}
    assert scores[10] == 1 / 61 + 1 / 62
    assert scores[20] == 1 / 62 + 1 / 61


def test_rrf_records_per_method_ranks():
    fused = reciprocal_rank_fusion({"bm25": [10, 20], "dense": [30]}, k=60)
    ranks = {f.item: f.ranks for f in fused}
    assert ranks[10] == {"bm25": 1}
    assert ranks[20] == {"bm25": 2}
    assert ranks[30] == {"dense": 1}


def test_dedup_keeps_one_chunk_per_doc():
    chunks = [
        Chunk(0, 0, "a", ""),
        Chunk(0, 1, "b", ""),
        Chunk(1, 0, "c", ""),
        Chunk(2, 0, "d", ""),
    ]
    index = FakeIndex(chunks, sparse=[0, 1, 2, 3], dense=[1, 0, 2, 3])
    hits = HybridRetriever(index).search("q", top_k=3)
    assert [h.chunk.doc_id for h in hits] == [0, 1, 2]


def test_empty_query_returns_empty_list():
    index = FakeIndex([Chunk(0, 0, "a", "")], sparse=[0], dense=[0])
    assert HybridRetriever(index).search("   ", top_k=5) == []


def test_hit_carries_source_ranks():
    chunks = [Chunk(0, 0, "a", ""), Chunk(1, 0, "b", "")]
    index = FakeIndex(chunks, sparse=[0, 1], dense=[1, 0])
    top = HybridRetriever(index).search("q", top_k=2)[0]
    assert top.bm25_rank == 1
    assert top.dense_rank == 2
