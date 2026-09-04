"""Benchmark the five retrieval configurations on the Russian MIRACL dev set.

Evaluate M dev queries against a corpus of all their qrels documents plus
random distractors drawn from the dev split, up to --subset documents. Only
the small dev download is needed, not the 9.5M passage corpus.
"""

from __future__ import annotations

import argparse
import platform
import random
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


def select_documents(all_docids, forced, subset_size: int, seed: int) -> list:
    """Pick the corpus: every qrels doc, padded with random distractors to N.

    The qrels documents must be indexed or their queries score zero for every
    method, so they go in first; if there are more of them than N, N is raised.
    """
    forced = set(forced)
    target = subset_size
    if len(forced) > target:
        print(f"relevant docs ({len(forced)}) exceed --subset {subset_size}; raising N")
        target = len(forced)

    rng = random.Random(seed)
    distractors = [d for d in all_docids if d not in forced]
    rng.shuffle(distractors)
    keep = list(forced) + distractors[: target - len(forced)]
    if len(keep) < target:
        print(f"only {len(keep)} unique dev passages available, below --subset {target}")
    return keep


def load_miracl_subset(num_queries: int, subset_size: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("miracl/miracl", "ru", split="dev", trust_remote_code=True)

    # every dev passage is a distractor candidate; sourcing them here keeps the
    # benchmark to the small dev download rather than the full corpus
    all_passages: dict[str, dict] = {}
    for row in ds:
        for p in row["positive_passages"] + row["negative_passages"]:
            all_passages.setdefault(p["docid"], {"title": p.get("title", ""), "text": p["text"]})

    selected = ds.select(range(min(num_queries, len(ds))))
    queries = []
    forced: set[str] = set()
    for row in selected:
        relevant = {p["docid"] for p in row["positive_passages"]}
        forced |= relevant
        queries.append({"query": row["query"], "relevant": relevant})

    keep = select_documents(all_passages.keys(), forced, subset_size, seed)
    passages = {docid: all_passages[docid] for docid in keep}
    sample = {"n_relevant": len(forced), "seed": seed}
    return passages, queries, sample


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
        f"- documents: {meta['documents']}",
        f"- relevant documents: {meta['relevant']}",
        f"- rng seed: {meta['seed']}",
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
    parser.add_argument("--queries", type=int, default=100, help="dev queries to evaluate")
    parser.add_argument("--subset", type=int, default=20000, help="target corpus size in documents")
    parser.add_argument("--seed", type=int, default=13, help="rng seed for distractor sampling")
    parser.add_argument("--out", default="benchmarks/results.md")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--no-dense", action="store_true", help="BM25 configs only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    passages, queries, sample = load_miracl_subset(args.queries, args.subset, args.seed)
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
        "documents": len(passages),
        "relevant": sample["n_relevant"],
        "seed": sample["seed"],
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
