"""Command line interface: index, search, bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ruhybrid.chunker import chunk_document
from ruhybrid.index import DEFAULT_MODEL, HybridIndex, IndexConfig
from ruhybrid.metrics import ndcg_at_k, recall_at_k
from ruhybrid.retriever import HybridRetriever


def _read_jsonl(path: str):
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


def cmd_index(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.input)
    chunks = []
    for doc_id, rec in enumerate(records):
        text = rec.get("text", "")
        if not text.strip():
            continue
        chunks.extend(
            chunk_document(
                doc_id=doc_id,
                text=text,
                title=rec.get("title", ""),
                source=rec.get("source"),
                max_chars=args.max_chars,
                overlap=args.overlap,
            )
        )
    if not chunks:
        raise SystemExit("no non-empty documents found in input")

    config = IndexConfig(
        ngram_size=args.ngram_size,
        dense=not args.no_dense,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    index = HybridIndex.build(chunks, config)
    index.save(args.out)
    print(f"indexed {len(records)} documents into {len(chunks)} chunks -> {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    index = HybridIndex.load(args.index, load_dense=not args.sparse_only)
    retriever = HybridRetriever(index)
    hits = retriever.search(args.query, top_k=args.top_k)
    if not hits:
        print("no results")
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. score={hit.score:.4f} bm25={hit.bm25_rank} dense={hit.dense_rank}")
        if hit.chunk.title:
            print(f"   title: {hit.chunk.title}")
        if hit.chunk.source:
            print(f"   source: {hit.chunk.source}")
        snippet = " ".join(hit.chunk.text[:200].split())
        print(f"   {snippet}")


def cmd_bench(args: argparse.Namespace) -> None:
    index = HybridIndex.load(args.index, load_dense=not args.sparse_only)
    retriever = HybridRetriever(index)
    qrels = _read_jsonl(args.qrels)

    recalls, ndcgs = [], []
    for rec in qrels:
        relevant = set(rec["relevant"])
        ranked = [h.chunk.doc_id for h in retriever.search(rec["query"], top_k=args.top_k)]
        recalls.append(recall_at_k(ranked, relevant, 100))
        ndcgs.append(ndcg_at_k(ranked, relevant, 10))

    n = len(qrels)
    print(f"queries:    {n}")
    print(f"Recall@100: {sum(recalls) / n:.4f}")
    print(f"nDCG@10:    {sum(ndcgs) / n:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruhybrid", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build an index from a JSONL corpus")
    p_index.add_argument("--input", required=True, help="corpus JSONL (text, title, source)")
    p_index.add_argument("--out", required=True, help="output index directory")
    p_index.add_argument("--ngram-size", type=int, default=4)
    p_index.add_argument("--no-dense", action="store_true", help="build BM25 only")
    p_index.add_argument("--model", default=DEFAULT_MODEL)
    p_index.add_argument("--batch-size", type=int, default=64)
    p_index.add_argument("--max-chars", type=int, default=1500)
    p_index.add_argument("--overlap", type=int, default=200)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="query an index")
    p_search.add_argument("--index", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--sparse-only", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_bench = sub.add_parser("bench", help="evaluate an index against a qrels file")
    p_bench.add_argument("--index", required=True)
    p_bench.add_argument("--qrels", required=True, help="JSONL with query and relevant doc ids")
    p_bench.add_argument("--top-k", type=int, default=100)
    p_bench.add_argument("--sparse-only", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
