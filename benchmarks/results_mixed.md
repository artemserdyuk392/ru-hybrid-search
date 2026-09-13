# MIRACL ru benchmark

- source: mteb/MIRACLReranking (ru)
- negatives: mixed
- queries: 997
- documents: 20000
- relevant documents: 2436
- queries with zero relevant in pool: 0
- rng seed: 13
- candidates per method: 200
- hardware: Darwin arm64
- python: 3.11.15
- index build time: 129.9 s
- eval time: 86.8 s

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.9997 | 0.8726 | 0.6970 | 0.6962 |
| bm25-stem+ngram | 0.9848 | 0.8208 | 0.6511 | 0.6510 |
| dense | 0.9997 | 0.9700 | 0.9068 | 0.8968 |
| hybrid-stem | 1.0000 | 0.9782 | 0.8467 | 0.8533 |
| hybrid-stem+ngram | 1.0000 | 0.9630 | 0.8351 | 0.8367 |

Recall@100 sits near 0.95-1.0 for every configuration in this pool and cannot tell them apart; rank the configurations by Recall@10, MRR@10 and nDCG@10 instead. Recall@100 is kept only to show the ceiling.

## nDCG@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0452 | [-0.0627, -0.0276] | <0.0001 | <0.0001 |
| dense | +0.2006 | [+0.1816, +0.2198] | <0.0001 | <0.0001 |
| hybrid-stem | +0.1571 | [+0.1447, +0.1698] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.1405 | [+0.1243, +0.1569] | <0.0001 | <0.0001 |

## Recall@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0518 | [-0.0716, -0.0317] | <0.0001 | <0.0001 |
| dense | +0.0975 | [+0.0810, +0.1143] | <0.0001 | <0.0001 |
| hybrid-stem | +0.1057 | [+0.0910, +0.1213] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.0904 | [+0.0747, +0.1064] | <0.0001 | <0.0001 |

## MRR@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0459 | [-0.0677, -0.0244] | <0.0001 | <0.0001 |
| dense | +0.2097 | [+0.1857, +0.2344] | <0.0001 | <0.0001 |
| hybrid-stem | +0.1497 | [+0.1332, +0.1667] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.1380 | [+0.1168, +0.1598] | <0.0001 | <0.0001 |

## Saturation

| config | nDCG@10 == 1.0 | Recall@100 == 1.0 |
| --- | --- | --- |
| bm25-stem | 0.2748 | 0.9990 |
| bm25-stem+ngram | 0.2447 | 0.9739 |
| dense | 0.5657 | 0.9990 |
| hybrid-stem | 0.4393 | 1.0000 |
| hybrid-stem+ngram | 0.4062 | 1.0000 |

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
| Recall@100 | 0.9997 | 0.9847 |  |  |  |
| Recall@10 | 0.8732 | 0.8202 | -0.0529 | [-0.0724, -0.0329] | <0.0001 |
| MRR@10 | 0.6984 | 0.6515 | -0.0469 | [-0.0682, -0.0255] | <0.0001 |
| nDCG@10 | 0.6972 | 0.6510 | -0.0463 | [-0.0636, -0.0287] | <0.0001 |

## Methodology and limitations

The pool is built from reranking candidates, not the full 9.5M passage corpus, so the task is easier than real MIRACL and the absolute values are inflated. MIRACL's hard negatives were chosen by a BM25-style retriever, so in the own-negatives pool every candidate is already a high-BM25-rank document for its query: BM25 can barely separate relevant from non-relevant there, while dense retrieval can, so that pool understates lexical methods (the n-gram trick included) and flatters dense. The n-gram bridge acts when a document is found, not when candidates are reranked, so it can only help in the other-negatives pool, where distractors are topically unrelated and lexical matching discriminates. Use --negatives own|other|mixed to switch. Comparing configurations within one pool is valid; comparing these numbers to published MIRACL results is not.

For Recall@10, MRR@10 and nDCG@10 each non-baseline configuration is compared to bm25-stem pairwise per query: delta is the mean difference, the 95% interval is a paired bootstrap over queries (10000 resamples, fixed seed) and p is a Wilcoxon signed-rank test. There are four comparisons per metric; p_holm applies a Holm-Bonferroni correction across those four and the raw p is kept beside it. The interval reflects only the spread across the queries in this pool, not generalization to other corpora.
