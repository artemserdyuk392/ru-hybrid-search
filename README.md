# ru-hybrid-search

![license](https://img.shields.io/badge/license-MIT-blue.svg)

Hybrid BM25 + dense retrieval for Russian text corpora. It builds a sparse
lexical index (BM25 over Snowball stems) and a dense index
(`intfloat/multilingual-e5-small` in FAISS), fuses them with Reciprocal Rank
Fusion, and ships a MIRACL benchmark that measures each configuration honestly,
including one idea that did not work.

The configurations are `bm25-stem`, `bm25-stem+ngram`, `dense`, and the two RRF
hybrids of them. The `+ngram` variant was an attempt to patch Russian morphology
in the lexical index by indexing character n-grams; the benchmark shows it does
not help on MIRACL, so n-grams are off by default (see "The n-gram option"). The
benchmark's main practical finding is that RRF fusion only helps when its two
components are close in strength.

## Benchmark

Headline is the `own` pool (997 dev queries, 99700 documents), the hardest and
least saturated of the three (see "Pool modes"). Full per-mode tables with
Recall@10, MRR@10, significance and the overlap slice are in
`benchmarks/results_own.md`, `results_mixed.md`, `results_other.md`;
`benchmarks/results.md` is the cross-mode summary.

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.9279 | 0.4556 | 0.3853 | 0.3471 |
| bm25-stem+ngram | 0.8920 | 0.4331 | 0.3641 | 0.3302 |
| dense | 0.9911 | 0.8156 | 0.7311 | 0.6903 |
| hybrid-stem | 0.9965 | 0.7014 | 0.5868 | 0.5491 |
| hybrid-stem+ngram | 0.9951 | 0.7059 | 0.5754 | 0.5425 |

Reproduce with `python benchmarks/run.py --negatives own` (or `mixed`, `other`);
`--smoke` runs a fast sparse-only check. Recall@100 is at the ceiling for every
configuration and cannot separate them; rank by Recall@10, MRR@10 and nDCG@10.

### Pool modes

The three `--negatives` modes differ only in what fills the pool around each
query's relevant passages, and that alone sets the difficulty. Baseline
bm25-stem nDCG@10:

- `own` 0.347: the query's own BM25-selected hard negatives (a reranking task).
  The pool is packed with documents already lexically close to the query.
- `mixed` 0.696 (default): half own hard negatives, half other queries'
  candidates.
- `other` 0.909: candidates belonging to other queries (a retrieval task). The
  distractors are topically unrelated, so lexical matching separates them easily.

## Install

```bash
pip install ru-hybrid-search
```

From source, in a virtualenv:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Dense retrieval pulls in sentence-transformers and faiss-cpu and downloads
`intfloat/multilingual-e5-small` on first use. For the benchmark also install
the extra: `pip install -e ".[bench]"`.

## Usage

Library, full hybrid index:

```python
from ruhybrid.chunker import chunk_document
from ruhybrid.index import HybridIndex, IndexConfig
from ruhybrid.retriever import HybridRetriever

chunks = chunk_document(0, "long document text ...", title="Title", source="url")
index = HybridIndex.build(chunks, IndexConfig())  # dense on, n-grams off
index.save("./idx")

index = HybridIndex.load("./idx")
for hit in HybridRetriever(index).search("your query", top_k=5):
    print(hit.score, hit.bm25_rank, hit.dense_rank, hit.chunk.title)
```

Command line. The corpus is JSONL with `text` and optional `title`/`source`:

```bash
ruhybrid index --input corpus.jsonl --out ./idx
ruhybrid search --index ./idx --query "как ускорить поиск" --top-k 5
ruhybrid bench --index ./idx --qrels qrels.jsonl
```

`--no-dense` on `index` builds a BM25-only index (no model needed); `--ngram-size
4` turns the n-gram option on. The qrels file for `bench` is JSONL, one object
per query: `{"query": "...", "relevant": [0, 4, 7]}`, where the ids are document
ids (the 0-based line number in the corpus).

## How it works

- Tokenization: lowercase, normalize ё to е, split on non-alphanumerics, drop
  nltk stopwords, then Snowball-stem each word using the stemmer for its
  alphabet. With `ngram_size > 0` it also emits character n-grams for Russian
  words of length >= 5 (the n-gram option, off by default).
- Sparse: `rank_bm25.BM25Okapi` over the tokenized chunks.
- Dense: `intfloat/multilingual-e5-small` with the required `passage: ` and
  `query: ` prefixes, normalized vectors in a FAISS `IndexFlatIP`, so inner
  product equals cosine.
- Fusion: Reciprocal Rank Fusion, `score = sum 1 / (k + rank)` with `k = 60`.
  RRF needs only ranks, so BM25 scores and cosine similarities never share a
  scale (Cormack, Clarke, Buettcher 2009). Results are then deduplicated by
  document id, keeping the best chunk of each document.

## The n-gram option (a negative result)

The premise was plausible. The Russian Snowball stemmer splits a verb and its
deverbal noun into different stems ("ускорить" -> "ускор", "ускорение" ->
"ускорен"), so a stem-only index cannot match one against the other. Character
4-grams of long Russian words share substrings across that split ("уско",
"скор") and could bridge it; enabling the option indexes those n-grams (marked
with a "#" prefix so an n-gram never collides with a real stem).

It does not help on MIRACL. `bm25-stem+ngram` loses to `bm25-stem` on nDCG@10 in
all three pools: -0.0452 in mixed and -0.0763 in other (both p<0.0001), and
-0.0168 in own (p=0.0785, not significant).

The gap the option targets barely exists in MIRACL. Of the 997 queries, 0 have
no lexical overlap with their positive documents and 3 have overlap at or below
0.25; the other 994 overlap heavily. There is almost nothing to bridge, and the
cost stays: the n-grams dilute IDF and add spurious matches, which drags the
score down. MIRACL queries were written by annotators who had read the passage,
so query and document nearly always share vocabulary.

Where the option could still pay off is corpora whose queries are written
independently of the documents: search logs, questions against a knowledge base,
support tickets. It stays in the code, off by default, for those cases.

## When the hybrid helps

Unweighted RRF works from ranks alone, so its result depends on how comparable
its inputs are. In `own`, dense (nDCG@10 0.690) is about twice bm25 (0.347) and
the hybrid lands at 0.549, below the better input. In `other`, dense (0.938) and
bm25 (0.909) are close and the hybrid reaches 0.951, above both. When one
component is roughly 1.5 to 2x the other, unweighted RRF drags the strong side
down toward the weak one. Measure bm25 and dense separately on your data first;
if they are that far apart, use weighted fusion or a reranker instead.

## Limitations

- On MIRACL the n-gram option lowers accuracy; it is off by default and only
  plausibly useful where queries do not share vocabulary with documents.
- Unweighted RRF can underperform its stronger component when the two are far
  apart in quality, as in the `own` and `mixed` pools above.
- All three benchmark pools are built from MIRACL reranking candidates, so every
  pool document is already a high-BM25-rank document for some query. That makes
  the pool harder than random for lexical methods and inflates absolute scores.
  Numbers are comparable between configurations within one mode and are not
  comparable to published MIRACL results.
- Dense search uses a brute-force FAISS `IndexFlatIP` that scans every vector.
  It is exact but does not scale to millions of chunks, and there is no
  quantization or IVF option yet. Building the dense index on CPU is slow.
- No incremental indexing. Adding or removing documents means rebuilding the
  whole index; `save`/`load` only spares you from re-embedding an unchanged
  corpus.
- English gets stemming only, no lemmatization. Mixed-script tokens are routed
  to one stemmer by a first-match alphabet check, which is a heuristic.
