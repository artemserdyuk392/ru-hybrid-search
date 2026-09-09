# MIRACL ru benchmark

Not run yet. Fill this in with `python benchmarks/run.py` (defaults: every usable
dev query, --negatives mixed, pool floor 20000 documents). The script overwrites
this file. Run `--negatives own` for the reranking pool and `--negatives other`
for the retrieval pool; only the other pool can show the n-gram effect. `--smoke`
is a fast sanity check and `--stats-only` recomputes every table below from
benchmarks/.cache/per_query.json without re-indexing.

- source: mteb/MIRACLReranking (ru)
- negatives: TODO
- queries: TODO
- documents: TODO
- relevant documents: TODO
- queries with zero relevant in pool: TODO
- rng seed: TODO
- candidates per method: 200
- hardware: TODO
- python: TODO

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | - | - | - | - |
| bm25-stem+ngram | - | - | - | - |
| dense | - | - | - | - |
| hybrid-stem | - | - | - | - |
| hybrid-stem+ngram | - | - | - | - |

Recall@100 sits near 0.95-1.0 for every configuration in this pool and cannot
tell them apart; rank the configurations by Recall@10, MRR@10 and nDCG@10
instead. Recall@100 is kept only to show the ceiling.

## nDCG@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | - | - | - | - |
| dense | - | - | - | - |
| hybrid-stem | - | - | - | - |
| hybrid-stem+ngram | - | - | - | - |

## Recall@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | - | - | - | - |
| dense | - | - | - | - |
| hybrid-stem | - | - | - | - |
| hybrid-stem+ngram | - | - | - | - |

## MRR@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | - | - | - | - |
| dense | - | - | - | - |
| hybrid-stem | - | - | - | - |
| hybrid-stem+ngram | - | - | - | - |

## Saturation

| config | nDCG@10 == 1.0 | Recall@100 == 1.0 |
| --- | --- | --- |
| bm25-stem | - | - |
| bm25-stem+ngram | - | - |
| dense | - | - |
| hybrid-stem | - | - |
| hybrid-stem+ngram | - | - |

If the bm25-stem share at nDCG@10 == 1.0 is above 0.5, the script prints and
records a warning that nDCG@10 is saturated and weak for comparing configs.

## Lexical-overlap slice

overlap = shared stemmed tokens between the query and its positive documents
divided by the query token count, tokenized without n-grams. Buckets: no-overlap
(overlap 0), low-overlap (0 to 0.25), rest (above 0.25). The n-gram bridge should
help most where overlap is low, so a whole-pool average washes the effect out.
Each bucket table pairs bm25-stem+ngram against bm25-stem; buckets under 20
queries show size only.

Queries with empty token sets: TODO (counted as overlap 0).

### no-overlap (n=TODO)

Filled per run: a table of Recall@100, Recall@10, MRR@10 and nDCG@10 for
bm25-stem and bm25-stem+ngram with the paired delta, 95% CI and p, or size only
when the bucket has fewer than 20 queries.

### low-overlap (n=TODO)

As above.

### rest (n=TODO)

As above.

## Methodology and limitations

The pool is built from reranking candidates, not the full 9.5M passage corpus,
so the task is easier than real MIRACL and the absolute values are inflated.
MIRACL's hard negatives were chosen by a BM25-style retriever, so in the
own-negatives pool every candidate is already a high-BM25-rank document for its
query: BM25 can barely separate relevant from non-relevant there, while dense
retrieval can, so that pool understates lexical methods (the n-gram trick
included) and flatters dense. The n-gram bridge acts when a document is found,
not when candidates are reranked, so it can only help in the other-negatives
pool, where distractors are topically unrelated and lexical matching
discriminates. Use --negatives own|other|mixed to switch. Comparing
configurations within one pool is valid; comparing these numbers to published
MIRACL results is not.

For Recall@10, MRR@10 and nDCG@10 each non-baseline configuration is compared to
bm25-stem pairwise per query: delta is the mean difference, the 95% interval is a
paired bootstrap over queries (10000 resamples, fixed seed) and p is a Wilcoxon
signed-rank test. There are four comparisons per metric; p_holm applies a
Holm-Bonferroni correction across those four and the raw p is kept beside it. The
interval reflects only the spread across the queries in this pool, not
generalization to other corpora.
