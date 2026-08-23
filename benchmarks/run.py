"""Benchmark the five retrieval configurations on the Russian MIRACL dev set.

The evaluation pool is the union of the positive and negative passages that
MIRACL ships inline with each dev query, so no multi-GB corpus download is
needed. This is a shared-pool benchmark, not full-corpus retrieval; see
results.md for what that means for the numbers.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from ruhybrid.chunker import chunk_document
from ruhybrid.index import DEFAULT_MODEL, HybridIndex, IndexConfig
from ruhybrid.metrics import ndcg_at_k, recall_at_k
from ruhybrid.retriever import reciprocal_rank_fusion

CONFIGS = [
    "bm25-stem",
    "bm25-stem+ngram",
    "dense",
    "hybrid-stem",
    "hybrid-stem+ngram",
]


def load_miracl_subset(subset: int | None):
    from datasets import load_dataset

    ds = load_dataset("miracl/miracl", "ru", split="dev", trust_remote_code=True)
    if subset:
        ds = ds.select(range(min(subset, len(ds))))

    passages: dict[str, dict] = {}
    queries = []
    for row in ds:
        relevant = set()
        for p in row["positive_passages"]:
            passages[p["docid"]] = {"title": p.get("title", ""), "text": p["text"]}
            relevant.add(p["docid"])
        for p in row["negative_passages"]:
            passages.setdefault(p["docid"], {"title": p.get("title", ""), "text": p["text"]})
        queries.append({"query": row["query"], "relevant": relevant})
    return passages, queries


def build_chunks(passages: dict[str, dict]):
    docids = list(passages.keys())
    docid_to_int = {d: i for i, d in enumerate(docids)}
    chunks = []
    for i, docid in enumerate(docids):
        p = passages[docid]
        chunks.extend(chunk_document(doc_id=i, text=p["text"], title=p["title"], source=docid))
    return chunks, docid_to_int


def build_indexes(chunks, dense: bool, model: str, batch_size: int):
    idx_stem = HybridIndex.build(chunks, IndexConfig(ngram_size=0, dense=False))
    idx_ngram = HybridIndex.build(chunks, IndexConfig(ngram_size=4, dense=False))
    idx_dense = None
    if dense:
        idx_dense = HybridIndex.build(
            chunks, IndexConfig(ngram_size=4, dense=True, model_name=model, batch_size=batch_size)
        )
    return idx_stem, idx_ngram, idx_dense


def _dedup_docs(index, chunk_ranking):
    seen = set()
    out = []
    for ci in chunk_ranking:
        doc_id = index.chunks[ci].doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def rankings_for_query(idx_stem, idx_ngram, idx_dense, query, candidates):
    stem_ci = idx_stem.search_sparse(query, candidates)
    ngram_ci = idx_ngram.search_sparse(query, candidates)
    out = {
        "bm25-stem": _dedup_docs(idx_stem, stem_ci),
        "bm25-stem+ngram": _dedup_docs(idx_ngram, ngram_ci),
    }
    if idx_dense is not None:
        dense_ci = idx_dense.search_dense(query, candidates)
        out["dense"] = _dedup_docs(idx_dense, dense_ci)
        hyb_stem = reciprocal_rank_fusion({"bm25": stem_ci, "dense": dense_ci})
        hyb_ngram = reciprocal_rank_fusion({"bm25": ngram_ci, "dense": dense_ci})
        out["hybrid-stem"] = _dedup_docs(idx_stem, [f.item for f in hyb_stem])
        out["hybrid-stem+ngram"] = _dedup_docs(idx_ngram, [f.item for f in hyb_ngram])
    return out


def evaluate(idx_stem, idx_ngram, idx_dense, queries, docid_to_int, candidates):
    sums = {c: [0.0, 0.0] for c in CONFIGS}
    counts = dict.fromkeys(CONFIGS, 0)
    for qr in queries:
        relevant = {docid_to_int[d] for d in qr["relevant"] if d in docid_to_int}
        ranks = rankings_for_query(idx_stem, idx_ngram, idx_dense, qr["query"], candidates)
        for config, ranked in ranks.items():
            sums[config][0] += recall_at_k(ranked, relevant, 100)
            sums[config][1] += ndcg_at_k(ranked, relevant, 10)
            counts[config] += 1
    results = {}
    for config in CONFIGS:
        if counts[config]:
            n = counts[config]
            results[config] = {"recall": sums[config][0] / n, "ndcg": sums[config][1] / n}
        else:
            results[config] = None
    return results


def write_results(path, results, meta):
    lines = [
        "# MIRACL ru benchmark",
        "",
        f"- queries: {meta['queries']}",
        f"- passages in pool: {meta['passages']}",
        f"- candidates per method: {meta['candidates']}",
        f"- hardware: {meta['hardware']}",
        f"- python: {meta['python']}",
        f"- index build time: {meta['build_time']:.1f} s",
        f"- eval time: {meta['eval_time']:.1f} s",
        "",
        "| config | Recall@100 | nDCG@10 |",
        "| --- | --- | --- |",
    ]
    for config in CONFIGS:
        r = results[config]
        if r is None:
            lines.append(f"| {config} | - | - |")
        else:
            lines.append(f"| {config} | {r['recall']:.4f} | {r['ndcg']:.4f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the MIRACL ru benchmark")
    parser.add_argument("--subset", type=int, default=None, help="limit to first N dev queries")
    parser.add_argument("--out", default="benchmarks/results.md")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--no-dense", action="store_true", help="BM25 configs only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    passages, queries = load_miracl_subset(args.subset)
    chunks, docid_to_int = build_chunks(passages)

    t0 = time.perf_counter()
    idx_stem, idx_ngram, idx_dense = build_indexes(
        chunks, dense=not args.no_dense, model=args.model, batch_size=args.batch_size
    )
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = evaluate(idx_stem, idx_ngram, idx_dense, queries, docid_to_int, args.candidates)
    eval_time = time.perf_counter() - t0

    meta = {
        "queries": len(queries),
        "passages": len(passages),
        "candidates": args.candidates,
        "hardware": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "build_time": build_time,
        "eval_time": eval_time,
    }
    write_results(args.out, results, meta)
    for config in CONFIGS:
        r = results[config]
        if r is None:
            print(f"{config:20s} skipped")
        else:
            print(f"{config:20s} Recall@100={r['recall']:.4f}  nDCG@10={r['ndcg']:.4f}")


if __name__ == "__main__":
    main()
