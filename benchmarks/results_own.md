# MIRACL ru benchmark

- source: mteb/MIRACLReranking (ru)
- negatives: own
- queries: 997
- documents: 99700
- relevant documents: 2436
- queries with zero relevant in pool: 0
- rng seed: 13
- candidates per method: 200
- hardware: Darwin arm64
- python: 3.11.15
- index build time: 631.5 s
- eval time: 460.1 s

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.9279 | 0.4556 | 0.3853 | 0.3471 |
| bm25-stem+ngram | 0.8920 | 0.4331 | 0.3641 | 0.3302 |
| dense | 0.9911 | 0.8156 | 0.7311 | 0.6903 |
| hybrid-stem | 0.9965 | 0.7014 | 0.5868 | 0.5491 |
| hybrid-stem+ngram | 0.9951 | 0.7059 | 0.5754 | 0.5425 |

Recall@100 sits near 0.95-1.0 for every configuration in this pool and cannot tell them apart; rank the configurations by Recall@10, MRR@10 and nDCG@10 instead. Recall@100 is kept only to show the ceiling.

## nDCG@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0168 | [-0.0339, +0.0001] | 0.0785 | 0.0785 |
| dense | +0.3433 | [+0.3194, +0.3664] | <0.0001 | <0.0001 |
| hybrid-stem | +0.2021 | [+0.1880, +0.2166] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.1954 | [+0.1767, +0.2138] | <0.0001 | <0.0001 |

## Recall@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0225 | [-0.0463, +0.0008] | 0.0628 | 0.0628 |
| dense | +0.3600 | [+0.3327, +0.3880] | <0.0001 | <0.0001 |
| hybrid-stem | +0.2458 | [+0.2234, +0.2691] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.2503 | [+0.2262, +0.2749] | <0.0001 | <0.0001 |

## MRR@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0212 | [-0.0425, -0.0004] | 0.0534 | 0.0534 |
| dense | +0.3458 | [+0.3171, +0.3744] | <0.0001 | <0.0001 |
| hybrid-stem | +0.2014 | [+0.1825, +0.2208] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.1900 | [+0.1663, +0.2136] | <0.0001 | <0.0001 |

## Saturation

| config | nDCG@10 == 1.0 | Recall@100 == 1.0 |
| --- | --- | --- |
| bm25-stem | 0.1113 | 0.8626 |
| bm25-stem+ngram | 0.1174 | 0.8265 |
| dense | 0.3180 | 0.9809 |
| hybrid-stem | 0.2006 | 0.9910 |
| hybrid-stem+ngram | 0.1886 | 0.9860 |

## Lexical-overlap slice

overlap = shared stemmed tokens between the query and its positive documents divided by the query token count, tokenized without n-grams. Buckets: no-overlap (overlap 0), low-overlap (0 to 0.25), rest (above 0.25). The n-gram bridge should help most where overlap is low, so a whole-pool average washes the effect out. Each bucket table pairs bm25-stem+ngram against bm25-stem; buckets under 20 queries show size only.

Queries with empty token sets: 0 (counted as overlap 0).

### no-overlap (n=0)

Statistically unusable (n < 20); size only.

### low-overlap (n=3)

Statistically unusable (n < 20); size only.

### rest (n=994)

| metric | bm25-stem | bm25-stem+ngram | delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Recall@100 | 0.9287 | 0.8917 |  |  |  |
| Recall@10 | 0.4565 | 0.4329 | -0.0236 | [-0.0467, +0.0002] | 0.0517 |
| MRR@10 | 0.3864 | 0.3646 | -0.0218 | [-0.0433, -0.0009] | 0.0471 |
| nDCG@10 | 0.3479 | 0.3304 | -0.0175 | [-0.0348, -0.0006] | 0.0676 |

## Methodology and limitations

The pool is built from reranking candidates, not the full 9.5M passage corpus, so the task is easier than real MIRACL and the absolute values are inflated. MIRACL's hard negatives were chosen by a BM25-style retriever, so in the own-negatives pool every candidate is already a high-BM25-rank document for its query: BM25 can barely separate relevant from non-relevant there, while dense retrieval can, so that pool understates lexical methods (the n-gram trick included) and flatters dense. The n-gram bridge acts when a document is found, not when candidates are reranked, so it can only help in the other-negatives pool, where distractors are topically unrelated and lexical matching discriminates. Use --negatives own|other|mixed to switch. Comparing configurations within one pool is valid; comparing these numbers to published MIRACL results is not.

For Recall@10, MRR@10 and nDCG@10 each non-baseline configuration is compared to bm25-stem pairwise per query: delta is the mean difference, the 95% interval is a paired bootstrap over queries (10000 resamples, fixed seed) and p is a Wilcoxon signed-rank test. There are four comparisons per metric; p_holm applies a Holm-Bonferroni correction across those four and the raw p is kept beside it. The interval reflects only the spread across the queries in this pool, not generalization to other corpora.
