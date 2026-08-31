"""Build the sparse (BM25) and dense (e5 + FAISS) indexes over chunks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ruhybrid.chunker import Chunk
from ruhybrid.tokenize import tokenize_document, tokenize_query

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@dataclass
class IndexConfig:
    ngram_size: int = 4
    dense: bool = True
    model_name: str = DEFAULT_MODEL
    batch_size: int = 64


class HybridIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenized: list[list[str]],
        bm25: BM25Okapi | None,
        config: IndexConfig,
        faiss_index=None,
    ):
        self.chunks = chunks
        self.tokenized = tokenized
        self.bm25 = bm25
        self.config = config
        self.faiss_index = faiss_index
        self._model = None

    @classmethod
    def build(cls, chunks: list[Chunk], config: IndexConfig | None = None) -> HybridIndex:
        config = config or IndexConfig()
        tokenized = [
            tokenize_document(c.text, config.ngram_size)
            for c in tqdm(chunks, desc="tokenize", unit="chunk")
        ]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        index = cls(chunks, tokenized, bm25, config)
        if config.dense:
            index._build_dense()
        return index

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        # e5 was trained with these prefixes; dropping them costs noticeable
        # retrieval quality. Vectors are normalized so inner product == cosine.
        model = self._load_model()
        embs = model.encode(
            [prefix + t for t in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embs.astype("float32")

    def _build_dense(self) -> None:
        import faiss

        # TODO: IndexFlatIP is a brute-force scan; switch to IVF or HNSW to
        # scale past roughly a million chunks.
        embs = self._encode([c.text for c in self.chunks], "passage: ")
        faiss_index = faiss.IndexFlatIP(embs.shape[1])
        faiss_index.add(embs)
        self.faiss_index = faiss_index

    def search_sparse(self, query: str, n: int) -> list[int]:
        if self.bm25 is None:
            return []
        tokens = tokenize_query(query, self.config.ngram_size)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:n]
        return [int(i) for i in order if scores[i] > 0]

    def search_dense(self, query: str, n: int) -> list[int]:
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        q = self._encode([query], "query: ")
        _, idx = self.faiss_index.search(q, min(n, self.faiss_index.ntotal))
        return [int(i) for i in idx[0] if i != -1]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        _write_json(p / "config.json", asdict(self.config))
        _write_json(p / "chunks.json", [c.to_dict() for c in self.chunks])
        # store tokens rather than pickling BM25: rebuilding the model on load
        # is cheap and avoids version-fragile pickles. The dense vectors, which
        # are expensive to recompute, go to the faiss file.
        _write_json(p / "tokenized.json", self.tokenized)
        if self.faiss_index is not None:
            import faiss

            faiss.write_index(self.faiss_index, str(p / "dense.faiss"))

    @classmethod
    def load(cls, path: str | Path, load_dense: bool = True) -> HybridIndex:
        p = Path(path)
        config = IndexConfig(**_read_json(p / "config.json"))
        chunks = [Chunk.from_dict(d) for d in _read_json(p / "chunks.json")]
        tokenized = _read_json(p / "tokenized.json")
        bm25 = BM25Okapi(tokenized) if tokenized else None
        faiss_index = None
        dense_path = p / "dense.faiss"
        if load_dense and dense_path.exists():
            import faiss

            faiss_index = faiss.read_index(str(dense_path))
        return cls(chunks, tokenized, bm25, config, faiss_index=faiss_index)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
