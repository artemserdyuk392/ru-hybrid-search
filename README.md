# ru-hybrid-search

![license](https://img.shields.io/badge/license-MIT-blue.svg)

Hybrid BM25 + dense retrieval for Russian text corpora. It fuses a sparse
lexical index with dense embeddings, because neither is enough on its own:
BM25 stumbles on Russian morphology, and dense retrieval is weak on rare
terms, exact names and numbers. On top of that, the library fixes a concrete
problem that a stemmer does not: the Snowball stemmer gives a Russian verb and
its deverbal noun different stems ("ускорить" -> "ускор", "ускорение" ->
"ускорен"), so a query for one does not match a document containing the other.
For long Russian words it also indexes character n-grams, and BM25 matches on
the shared n-grams ("уско", "скор"), bridging the gap.

## The problem, concretely

```python
from ruhybrid.chunker import chunk_document
from ruhybrid.index import HybridIndex, IndexConfig
from ruhybrid.retriever import HybridRetriever

docs = [
    "Статья про ускорение поиска по большому индексу.",
    "Рецепт борща со свеклой и капустой.",
    "Как повысить надежность распределенных баз данных.",
]
chunks = []
for doc_id, text in enumerate(docs):
    chunks.extend(chunk_document(doc_id, text, title=f"doc {doc_id}"))

# dense=False keeps this snippet runnable without downloading a model;
# the n-gram bridge is a pure BM25 feature and works on its own.
index = HybridIndex.build(chunks, IndexConfig(dense=False))
hits = HybridRetriever(index).search("как ускорить поиск", top_k=1)
print(hits[0].chunk.text)
# -> Статья про ускорение поиска по большому индексу.
```

A plain stemmed BM25 index returns nothing here: "ускорить" stems to "ускор"
and "ускорение" to "ускорен", which never match. The shared 4-grams do.

## Benchmark

Recall@10, MRR@10 and nDCG@10 on a Russian MIRACL dev sample, five
configurations. Recall@100 is reported too, but it sits near 1.0 for every
configuration and cannot separate them. Not run yet. Reproduce with `python
benchmarks/run.py`; `--smoke` runs a fast sparse-only check. The script writes
the full tables, per-metric significance and timings into `benchmarks/results.md`.

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | - | - | - | - |
| bm25-stem+ngram | - | - | - | - |
| dense | - | - | - | - |
| hybrid-stem | - | - | - | - |
| hybrid-stem+ngram | - | - | - | - |

Data comes from `mteb/MIRACLReranking` (ru), which ships each query's reranking
candidates as parquet, so no multi-GB corpus download is needed. `--negatives`
chooses the pool: `own` keeps each query's own BM25-selected hard negatives (a
reranking task), `other` replaces them with candidates from other queries (a
retrieval task), and `mixed` (default) blends the two. The direction of the bias
matters: because the hard negatives are already high-BM25-rank documents for
their query, BM25 can barely separate relevant from non-relevant in the `own`
pool, so that pool understates lexical methods and flatters dense. The n-gram
bridge works when a document is found, not when candidates are reranked, so it
can only show an effect in the `other` pool. None of these pools is the full
9.5M corpus, so the task is easier than real MIRACL and absolute numbers are
inflated; compare configurations within one pool, not against published MIRACL.

For Recall@10, MRR@10 and nDCG@10, results.md reports each configuration's
paired per-query difference against bm25-stem with a paired-bootstrap 95%
interval, a Wilcoxon p, and a Holm-Bonferroni-corrected p_holm across the four
comparisons. Those intervals reflect only the spread across these queries, not
generalization to other corpora.

Because whole-pool averages hide the n-gram effect, results.md also slices
queries by query/document lexical overlap (no-overlap, low-overlap, rest) and
reports the n-gram-vs-stem comparison per bucket, where the effect should be
strongest at low overlap. It also records how often each metric hits its
ceiling and warns when nDCG@10 is saturated.

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
index = HybridIndex.build(chunks, IndexConfig())  # dense on by default
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

`--no-dense` on `index` builds a BM25-only index (no model needed).
The qrels file for `bench` is JSONL, one object per query:
`{"query": "...", "relevant": [0, 4, 7]}`, where the ids are document ids
(the 0-based line number in the corpus).

## How it works

- Tokenization: lowercase, normalize ё to е, split on non-alphanumerics, drop
  nltk stopwords, then Snowball-stem each word using the stemmer for its
  alphabet. For Russian words of length >= 5 it additionally emits character
  n-grams (default size 4). Each n-gram is prefixed with a "#" marker before it
  enters the index. Without the marker a 4-gram could equal the stem of some
  short word, so a query n-gram would match that unrelated stem and fire a false
  BM25 hit. The prefix keeps the n-gram space and the stem space disjoint, so an
  n-gram only ever matches another n-gram.
- Sparse: `rank_bm25.BM25Okapi` over the tokenized chunks.
- Dense: `intfloat/multilingual-e5-small` with the required `passage: ` and
  `query: ` prefixes, normalized vectors in a FAISS `IndexFlatIP`, so inner
  product equals cosine.
- Fusion: Reciprocal Rank Fusion, `score = sum 1 / (k + rank)` with `k = 60`.
  RRF needs only ranks, so BM25 scores and cosine similarities never share a
  scale (Cormack, Clarke, Buettcher 2009). Results are then deduplicated by
  document id, keeping the best chunk of each document.

## Limitations

- The n-gram bridge is lexical, not semantic. It also links unrelated words
  that happen to share 4-grams, and it enlarges the sparse index noticeably on
  corpora full of long Russian words.
- Dense search uses a brute-force FAISS `IndexFlatIP` that scans every vector.
  It is exact but does not scale to millions of chunks, and there is no
  quantization or IVF option yet. Building the dense index on CPU is slow.
- No incremental indexing. Adding or removing documents means rebuilding the
  whole index; `save`/`load` only spares you from re-embedding an unchanged
  corpus.
- English gets stemming only, no lemmatization. Mixed-script tokens are routed
  to one stemmer by a first-match alphabet check, which is a heuristic.
