"""Benchmark the five retrieval configurations on the Russian MIRACL dev set.

Source is mteb/MIRACLReranking (parquet, loads on datasets 5.x). --negatives
picks the pool: own (reranking), other (retrieval), or mixed. See results.md
for the methodology caveats and what each pool does to the metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from ruhybrid.chunker import chunk_document
from ruhybrid.index import DEFAULT_MODEL, HybridIndex, IndexConfig
from ruhybrid.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from ruhybrid.retriever import reciprocal_rank_fusion
from ruhybrid.tokenize import tokenize_document, tokenize_query

DATASET = "mteb/MIRACLReranking"
LANG = "ru"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
PER_QUERY_CACHE = CACHE_DIR / "per_query.json"

CONFIGS = [
    "bm25-stem",
    "bm25-stem+ngram",
    "dense",
    "hybrid-stem",
    "hybrid-stem+ngram",
]
BASELINE = "bm25-stem"

# metric key -> label; all four are reported as means, the paired ones also
# get a per-query significance test against the baseline
ALL_METRICS = [("recall100", "Recall@100"), ("recall10", "Recall@10"),
               ("mrr10", "MRR@10"), ("ndcg10", "nDCG@10")]
PAIRED_METRICS = [("ndcg10", "nDCG@10"), ("recall10", "Recall@10"), ("mrr10", "MRR@10")]

BOOTSTRAP_SEED = 0
BOOTSTRAP_RESAMPLES = 10000

# the overlap slice isolates the n-gram effect; buckets below this size are
# too small for a meaningful paired test
MIN_BUCKET_QUERIES = 20
SATURATION_THRESHOLD = 0.5
BUCKET_ORDER = ["no-overlap", "low-overlap", "rest"]
BUCKET_CONFIGS = ["bm25-stem", "bm25-stem+ngram"]


def _download(refresh: bool):
    from datasets import load_dataset

    mode = "force_redownload" if refresh else None
    queries = load_dataset(DATASET, f"{LANG}-queries", split="dev", download_mode=mode)
    qrels = load_dataset(DATASET, f"{LANG}-qrels", split="dev", download_mode=mode)

    query_text = {r["_id"]: r["text"] for r in queries}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qrels:
        qrels_by_query[r["query-id"]][r["corpus-id"]] = int(r["score"])
    return query_text, dict(qrels_by_query)


def select_pool(query_text, qrels_by_query, num_queries, subset_size, seed, negatives):
    """Pick queries and their document pool.

    Only queries with a judged-relevant candidate are usable. Positives of the
    selected queries are always in the pool; the distractors depend on the mode:
    own uses each query's own hard negatives (reranking), other uses candidates
    of non-selected queries (retrieval), mixed takes half of each. A query's
    own positives are never let in as distractors.
    """
    usable = sorted(q for q, cands in qrels_by_query.items() if any(s > 0 for s in cands.values()))
    selected = usable if num_queries is None else usable[:num_queries]
    selected_set = set(selected)

    queries = []
    for qid in selected:
        pos = sorted(d for d, s in qrels_by_query[qid].items() if s > 0)
        queries.append({"query_id": qid, "query": query_text[qid], "relevant": pos})
    all_positives = {d for q in queries for d in q["relevant"]}

    own_neg = [
        d for qid in selected for d, s in qrels_by_query[qid].items() if s == 0
    ]
    other_cand = [
        d for qid, cands in qrels_by_query.items() if qid not in selected_set for d in cands
    ]
    # guard: a selected query's positive must never enter the pool as a distractor
    own_neg = [d for d in own_neg if d not in all_positives]
    other_cand = [d for d in other_cand if d not in all_positives]

    rng = random.Random(seed)
    pool = set(all_positives)
    if negatives == "own":
        pool.update(own_neg)
    elif negatives == "other":
        rng.shuffle(other_cand)
        pool.update(other_cand[: max(0, subset_size - len(pool))])
    else:  # mixed
        rng.shuffle(own_neg)
        rng.shuffle(other_cand)
        half = max(0, subset_size - len(pool)) // 2
        pool.update(own_neg[:half])
        pool.update(other_cand[: max(0, subset_size - len(pool))])
    return pool, queries


def _corpus_text(pool: set, refresh: bool):
    from datasets import load_dataset

    mode = "force_redownload" if refresh else None
    corpus = load_dataset(DATASET, f"{LANG}-corpus", split="dev", download_mode=mode)
    passages = {}
    for row in corpus:
        if row["_id"] in pool:
            passages[row["_id"]] = {"title": row["title"] or "", "text": row["text"]}
    return passages


def load_pool(num_queries, subset_size, seed, negatives, refresh):
    qtag = "all" if num_queries is None else str(num_queries)
    cache = CACHE_DIR / f"pool_{LANG}_q{qtag}_n{subset_size}_s{seed}_{negatives}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["passages"], data["queries"]

    query_text, qrels_by_query = _download(refresh)
    pool, queries = select_pool(
        query_text, qrels_by_query, num_queries, subset_size, seed, negatives
    )
    passages = _corpus_text(pool, refresh)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"passages": passages, "queries": queries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return passages, queries


def diagnostics(passages, queries) -> dict:
    zero = sum(1 for q in queries if not any(d in passages for d in q["relevant"]))
    return {
        "queries": len(queries),
        "documents": len(passages),
        "relevant": sum(len(q["relevant"]) for q in queries),
        "zero_relevant": zero,
    }


def query_overlaps(queries, passages):
    """Lexical overlap between each query and its positive documents.

    overlap = shared stemmed tokens (no n-grams) divided by the query token
    count. It is a proxy for how much a query can be answered by direct lexical
    match; the n-gram bridge is expected to matter only where overlap is low.
    """
    overlaps = []
    empty = 0
    for qr in queries:
        q_tokens = set(tokenize_query(qr["query"], ngram_size=0))
        if not q_tokens:
            empty += 1
            overlaps.append(0.0)
            continue
        d_tokens = set()
        for docid in qr["relevant"]:
            passage = passages.get(docid)
            if passage:
                d_tokens.update(tokenize_document(passage["text"], ngram_size=0))
        overlaps.append(len(q_tokens & d_tokens) / len(q_tokens))
    return overlaps, empty


def build_chunks(passages: dict):
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
    """Return per-query metric arrays for each configuration.

    Keeping the raw per-query values (not just the mean) is what lets the
    significance test run later without re-indexing.
    """
    per_query = {c: {m: [] for m, _ in ALL_METRICS} for c in CONFIGS}
    for qr in queries:
        relevant = {docid_to_int[d] for d in qr["relevant"] if d in docid_to_int}
        ranks = rankings_for_query(idx_stem, idx_ngram, idx_dense, qr["query"], candidates)
        for config, ranked in ranks.items():
            per_query[config]["recall100"].append(recall_at_k(ranked, relevant, 100))
            per_query[config]["recall10"].append(recall_at_k(ranked, relevant, 10))
            per_query[config]["mrr10"].append(mrr_at_k(ranked, relevant, 10))
            per_query[config]["ndcg10"].append(ndcg_at_k(ranked, relevant, 10))
    return per_query


def bootstrap_ci(diff, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, alpha=0.05):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    means = diff[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def wilcoxon_p(config_scores, baseline_scores):
    a = np.asarray(config_scores, dtype=float)
    b = np.asarray(baseline_scores, dtype=float)
    # Wilcoxon is undefined when every pair is tied; report no difference
    if len(a) == 0 or np.all(a == b):
        return 1.0
    from scipy.stats import wilcoxon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(wilcoxon(a, b).pvalue)
        except ValueError:
            return 1.0


def holm_bonferroni(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted = {}
    running = 0.0
    for i, name in enumerate(order):
        running = max(running, (m - i) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def _mean(arr):
    return float(np.mean(arr)) if arr else None


def compute_stats(per_query: dict):
    means = {c: {m: _mean(per_query[c][m]) for m, _ in ALL_METRICS} for c in CONFIGS}
    paired = {}
    for key, _label in PAIRED_METRICS:
        base = per_query[BASELINE][key]
        comps = {}
        for config in CONFIGS:
            if config == BASELINE:
                continue
            arr = per_query[config][key]
            if not arr or len(arr) != len(base):
                continue
            diff = np.asarray(arr, dtype=float) - np.asarray(base, dtype=float)
            comps[config] = {
                "delta": float(diff.mean()),
                "ci": bootstrap_ci(diff),
                "p": wilcoxon_p(arr, base),
            }
        adjusted = holm_bonferroni({c: comps[c]["p"] for c in comps})
        for config in comps:
            comps[config]["p_holm"] = adjusted[config]
        paired[key] = comps
    return means, paired


def compute_saturation(per_query: dict) -> dict:
    sat = {}
    for config in CONFIGS:
        ndcg = per_query[config]["ndcg10"]
        recall100 = per_query[config]["recall100"]
        sat[config] = {
            "ndcg1": (sum(x == 1.0 for x in ndcg) / len(ndcg)) if ndcg else None,
            "recall1": (sum(x == 1.0 for x in recall100) / len(recall100)) if recall100 else None,
        }
    return sat


def saturation_warning(saturation) -> str | None:
    base = saturation[BASELINE]["ndcg1"]
    if base is not None and base > SATURATION_THRESHOLD:
        return (
            f"WARNING: bm25-stem reaches nDCG@10 == 1.0 on {base:.2f} of queries; "
            "nDCG@10 is saturated here and comparing configurations by it is weak."
        )
    return None


def _assign_buckets(overlaps):
    buckets = {name: [] for name in BUCKET_ORDER}
    for i, overlap in enumerate(overlaps):
        if overlap == 0.0:
            buckets["no-overlap"].append(i)
        elif overlap <= 0.25:
            buckets["low-overlap"].append(i)
        else:
            buckets["rest"].append(i)
    return buckets


def _slice(arr, idx, n_total):
    if len(arr) != n_total:
        return None
    return [arr[i] for i in idx]


def compute_buckets(per_query: dict, overlaps) -> dict:
    n_total = len(overlaps)
    assigned = _assign_buckets(overlaps)
    out = {}
    for name in BUCKET_ORDER:
        idx = assigned[name]
        entry = {
            "size": len(idx),
            "usable": len(idx) >= MIN_BUCKET_QUERIES,
            "means": {},
            "paired": {},
        }
        if entry["usable"]:
            for config in BUCKET_CONFIGS:
                entry["means"][config] = {
                    key: _mean(_slice(per_query[config][key], idx, n_total))
                    for key, _ in ALL_METRICS
                }
            for key, _label in PAIRED_METRICS:
                a = _slice(per_query["bm25-stem+ngram"][key], idx, n_total)
                b = _slice(per_query[BASELINE][key], idx, n_total)
                if a is None or b is None:
                    continue
                diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
                entry["paired"][key] = {
                    "delta": float(diff.mean()),
                    "ci": bootstrap_ci(diff),
                    "p": wilcoxon_p(a, b),
                }
        out[name] = entry
    return out


_METHODOLOGY = (
    "The pool is built from reranking candidates, not the full 9.5M passage "
    "corpus, so the task is easier than real MIRACL and the absolute values are "
    "inflated. MIRACL's hard negatives were chosen by a BM25-style retriever, so "
    "in the own-negatives pool every candidate is already a high-BM25-rank "
    "document for its query: BM25 can barely separate relevant from non-relevant "
    "there, while dense retrieval can, so that pool understates lexical methods "
    "(the n-gram trick included) and flatters dense. The n-gram bridge acts when "
    "a document is found, not when candidates are reranked, so it can only help "
    "in the other-negatives pool, where distractors are topically unrelated and "
    "lexical matching discriminates. Use --negatives own|other|mixed to switch. "
    "Comparing configurations within one pool is valid; comparing these numbers "
    "to published MIRACL results is not."
)

_CEILING_NOTE = (
    "Recall@100 sits near 0.95-1.0 for every configuration in this pool and "
    "cannot tell them apart; rank the configurations by Recall@10, MRR@10 and "
    "nDCG@10 instead. Recall@100 is kept only to show the ceiling."
)

_STATS_NOTE = (
    "For Recall@10, MRR@10 and nDCG@10 each non-baseline configuration is "
    "compared to bm25-stem pairwise per query: delta is the mean difference, the "
    "95% interval is a paired bootstrap over queries (10000 resamples, fixed "
    "seed) and p is a Wilcoxon signed-rank test. There are four comparisons per "
    "metric; p_holm applies a Holm-Bonferroni correction across those four and "
    "the raw p is kept beside it. The interval reflects only the spread across "
    "the queries in this pool, not generalization to other corpora."
)

_OVERLAP_NOTE = (
    "overlap = shared stemmed tokens between the query and its positive "
    "documents divided by the query token count, tokenized without n-grams. "
    "Buckets: no-overlap (overlap 0), low-overlap (0 to 0.25), rest (above "
    "0.25). The n-gram bridge should help most where overlap is low, so a "
    "whole-pool average washes the effect out. Each bucket table pairs "
    "bm25-stem+ngram against bm25-stem; buckets under 20 queries show size only."
)


def _fmt(value):
    return f"{value:.4f}" if value is not None else "-"


def _fmt_p(value):
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def write_results(path, means, paired, saturation, buckets, meta):
    lines = [
        "# MIRACL ru benchmark",
        "",
        f"- source: {DATASET} ({LANG})",
        f"- negatives: {meta['negatives']}",
        f"- queries: {meta['queries']}",
        f"- documents: {meta['documents']}",
        f"- relevant documents: {meta['relevant']}",
        f"- queries with zero relevant in pool: {meta['zero_relevant']}",
        f"- rng seed: {meta['seed']}",
        f"- candidates per method: {meta['candidates']}",
        f"- hardware: {meta['hardware']}",
        f"- python: {meta['python']}",
        f"- index build time: {meta['build_time']:.1f} s",
        f"- eval time: {meta['eval_time']:.1f} s",
        "",
        "| config | " + " | ".join(label for _, label in ALL_METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in ALL_METRICS) + " |",
    ]
    for config in CONFIGS:
        cells = " | ".join(_fmt(means[config][key]) for key, _ in ALL_METRICS)
        lines.append(f"| {config} | {cells} |")
    lines += ["", _CEILING_NOTE, ""]

    for key, label in PAIRED_METRICS:
        lines += [
            f"## {label} vs bm25-stem",
            "",
            "| config | delta | 95% CI | p | p_holm |",
            "| --- | --- | --- | --- | --- |",
        ]
        comps = paired[key]
        for config in CONFIGS:
            if config == BASELINE or config not in comps:
                continue
            s = comps[config]
            ci = f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]"
            lines.append(
                f"| {config} | {s['delta']:+.4f} | {ci} | "
                f"{_fmt_p(s['p'])} | {_fmt_p(s['p_holm'])} |"
            )
        lines.append("")

    lines += [
        "## Saturation",
        "",
        "| config | nDCG@10 == 1.0 | Recall@100 == 1.0 |",
        "| --- | --- | --- |",
    ]
    for config in CONFIGS:
        s = saturation[config]
        lines.append(f"| {config} | {_fmt(s['ndcg1'])} | {_fmt(s['recall1'])} |")
    warning = saturation_warning(saturation)
    if warning:
        lines += ["", warning]
    lines.append("")

    lines += [
        "## Lexical-overlap slice",
        "",
        _OVERLAP_NOTE,
        "",
        f"Queries with empty token sets: {meta.get('empty_query_tokens', 0)} "
        "(counted as overlap 0).",
        "",
    ]
    for name in BUCKET_ORDER:
        bucket = buckets[name]
        lines += [f"### {name} (n={bucket['size']})", ""]
        if not bucket["usable"]:
            lines += [f"Statistically unusable (n < {MIN_BUCKET_QUERIES}); size only.", ""]
            continue
        lines += [
            "| metric | bm25-stem | bm25-stem+ngram | delta | 95% CI | p |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, label in ALL_METRICS:
            stem = _fmt(bucket["means"]["bm25-stem"][key])
            ngram = _fmt(bucket["means"]["bm25-stem+ngram"][key])
            pair = bucket["paired"].get(key)
            if pair:
                ci = f"[{pair['ci'][0]:+.4f}, {pair['ci'][1]:+.4f}]"
                lines.append(
                    f"| {label} | {stem} | {ngram} | "
                    f"{pair['delta']:+.4f} | {ci} | {_fmt_p(pair['p'])} |"
                )
            else:
                lines.append(f"| {label} | {stem} | {ngram} |  |  |  |")
        lines.append("")

    lines += ["## Methodology and limitations", "", _METHODOLOGY, "", _STATS_NOTE]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_stats(means, paired, saturation, buckets, meta):
    for config in CONFIGS:
        m = means[config]
        if m["ndcg10"] is None:
            print(f"{config:20s} skipped")
            continue
        print(
            f"{config:20s} R@100={m['recall100']:.4f}  R@10={m['recall10']:.4f}  "
            f"MRR@10={m['mrr10']:.4f}  nDCG@10={m['ndcg10']:.4f}"
        )
    for key, label in PAIRED_METRICS:
        shown = [c for c in CONFIGS if c != BASELINE and c in paired[key]]
        if not shown:
            continue
        print(f"  {label} vs bm25-stem:")
        for config in shown:
            s = paired[key][config]
            print(
                f"    {config:20s} delta={s['delta']:+.4f}  "
                f"CI=[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]  "
                f"p={_fmt_p(s['p'])}  p_holm={_fmt_p(s['p_holm'])}"
            )

    print("saturation (share of queries at the ceiling):")
    for config in CONFIGS:
        s = saturation[config]
        if s["ndcg1"] is None:
            print(f"  {config:20s} skipped")
            continue
        print(f"  {config:20s} nDCG@10==1.0: {s['ndcg1']:.3f}  Recall@100==1.0: {s['recall1']:.3f}")
    warning = saturation_warning(saturation)
    if warning:
        print(warning)

    print(f"overlap buckets (empty-token queries: {meta.get('empty_query_tokens', 0)}):")
    for name in BUCKET_ORDER:
        bucket = buckets[name]
        if not bucket["usable"]:
            print(f"  {name}: n={bucket['size']} (statistically unusable, n<{MIN_BUCKET_QUERIES})")
            continue
        print(f"  {name}: n={bucket['size']}")
        for key, label in PAIRED_METRICS:
            pair = bucket["paired"].get(key)
            if pair:
                print(
                    f"    {label:9s} ngram-stem delta={pair['delta']:+.4f}  "
                    f"CI=[{pair['ci'][0]:+.4f}, {pair['ci'][1]:+.4f}]  p={_fmt_p(pair['p'])}"
                )


def _queries_arg(value):
    if value in ("all", "0"):
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer, 0, or 'all'") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1, or 0/all for every usable query")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the MIRACL ru benchmark")
    parser.add_argument(
        "--queries", type=_queries_arg, default=None,
        help="dev queries to evaluate; 0 or 'all' (default) means every usable query",
    )
    parser.add_argument(
        "--negatives", choices=["own", "other", "mixed"], default="mixed",
        help="distractor source: own (reranking), other (retrieval), mixed (default)",
    )
    parser.add_argument("--subset", type=int, default=20000, help="floor for the pool size")
    parser.add_argument("--seed", type=int, default=13, help="rng seed for distractor selection")
    parser.add_argument("--out", default="benchmarks/results.md")
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--no-dense", action="store_true", help="BM25 configs only")
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    parser.add_argument("--smoke", action="store_true", help="5 queries, 500 docs, sparse only")
    parser.add_argument(
        "--stats-only", action="store_true",
        help="recompute the tables from per_query.json without re-indexing",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    if args.stats_only:
        cached = json.loads(PER_QUERY_CACHE.read_text(encoding="utf-8"))
        per_query, overlaps, meta = cached["per_query"], cached.get("overlaps", []), cached["meta"]
        means, paired = compute_stats(per_query)
        saturation = compute_saturation(per_query)
        buckets = compute_buckets(per_query, overlaps)
        write_results(args.out, means, paired, saturation, buckets, meta)
        _print_stats(means, paired, saturation, buckets, meta)
        return

    if args.smoke:
        args.queries, args.subset, args.no_dense = 5, 500, True

    passages, queries = load_pool(
        args.queries, args.subset, args.seed, args.negatives, args.refresh
    )

    diag = diagnostics(passages, queries)
    print(
        f"pool ({args.negatives}): {diag['queries']} queries, {diag['documents']} documents, "
        f"{diag['relevant']} relevant, {diag['zero_relevant']} queries with zero relevant, "
        f"seed {args.seed}"
    )
    if diag["zero_relevant"] > 0:
        raise SystemExit(
            f"{diag['zero_relevant']} queries have no relevant document in the pool; "
            "metrics would be invalid. Re-run with --refresh or a larger --subset."
        )

    overlaps, empty_query_tokens = query_overlaps(queries, passages)

    chunks, docid_to_int = build_chunks(passages)

    t0 = time.perf_counter()
    idx_stem, idx_ngram, idx_dense = build_indexes(
        chunks, dense=not args.no_dense, model=args.model, batch_size=args.batch_size
    )
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_query = evaluate(idx_stem, idx_ngram, idx_dense, queries, docid_to_int, args.candidates)
    eval_time = time.perf_counter() - t0

    meta = {
        **diag,
        "negatives": args.negatives,
        "seed": args.seed,
        "candidates": args.candidates,
        "empty_query_tokens": empty_query_tokens,
        "hardware": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "build_time": build_time,
        "eval_time": eval_time,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PER_QUERY_CACHE.write_text(
        json.dumps(
            {"per_query": per_query, "overlaps": overlaps, "meta": meta}, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    means, paired = compute_stats(per_query)
    saturation = compute_saturation(per_query)
    buckets = compute_buckets(per_query, overlaps)
    write_results(args.out, means, paired, saturation, buckets, meta)
    _print_stats(means, paired, saturation, buckets, meta)


if __name__ == "__main__":
    main()
