# MIRACL ru benchmark

- source: mteb/MIRACLReranking (ru)
- negatives: other
- queries: 997
- documents: 20000
- relevant documents: 2436
- queries with zero relevant in pool: 0
- rng seed: 13
- candidates per method: 200
- hardware: Darwin arm64
- python: 3.11.15
- index build time: 127.1 s
- eval time: 89.1 s

| config | Recall@100 | Recall@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.9992 | 0.9779 | 0.9040 | 0.9087 |
| bm25-stem+ngram | 0.9880 | 0.9030 | 0.8424 | 0.8325 |
| dense | 0.9997 | 0.9806 | 0.9438 | 0.9377 |
| hybrid-stem | 1.0000 | 0.9946 | 0.9471 | 0.9512 |
| hybrid-stem+ngram | 1.0000 | 0.9842 | 0.9263 | 0.9279 |

Recall@100 sits near 0.95-1.0 for every configuration in this pool and cannot tell them apart; rank the configurations by Recall@10, MRR@10 and nDCG@10 instead. Recall@100 is kept only to show the ceiling.

## nDCG@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0763 | [-0.0928, -0.0602] | <0.0001 | <0.0001 |
| dense | +0.0290 | [+0.0169, +0.0415] | <0.0001 | <0.0001 |
| hybrid-stem | +0.0425 | [+0.0343, +0.0511] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.0192 | [+0.0073, +0.0310] | 0.0085 | 0.0085 |

## Recall@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0749 | [-0.0913, -0.0592] | <0.0001 | <0.0001 |
| dense | +0.0027 | [-0.0065, +0.0121] | 0.6756 | 0.6756 |
| hybrid-stem | +0.0167 | [+0.0100, +0.0241] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.0063 | [-0.0019, +0.0147] | 0.2359 | 0.4719 |

## MRR@10 vs bm25-stem

| config | delta | 95% CI | p | p_holm |
| --- | --- | --- | --- | --- |
| bm25-stem+ngram | -0.0616 | [-0.0805, -0.0432] | <0.0001 | <0.0001 |
| dense | +0.0398 | [+0.0243, +0.0557] | <0.0001 | <0.0001 |
| hybrid-stem | +0.0431 | [+0.0323, +0.0543] | <0.0001 | <0.0001 |
| hybrid-stem+ngram | +0.0223 | [+0.0069, +0.0379] | 0.0025 | 0.0025 |

## Saturation

| config | nDCG@10 == 1.0 | Recall@100 == 1.0 |
| --- | --- | --- |
| bm25-stem | 0.6861 | 0.9980 |
| bm25-stem+ngram | 0.5797 | 0.9809 |
| dense | 0.7212 | 0.9990 |
| hybrid-stem | 0.7854 | 1.0000 |
| hybrid-stem+ngram | 0.7081 | 1.0000 |

WARNING: bm25-stem reaches nDCG@10 == 1.0 on 0.69 of queries; nDCG@10 is saturated here and comparing configurations by it is weak.

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
| Recall@100 | 0.9992 | 0.9879 |  |  |  |
| Recall@10 | 0.9778 | 0.9027 | -0.0752 | [-0.0909, -0.0592] | <0.0001 |
| MRR@10 | 0.9049 | 0.8419 | -0.0629 | [-0.0815, -0.0443] | <0.0001 |
| nDCG@10 | 0.9093 | 0.8319 | -0.0774 | [-0.0936, -0.0613] | <0.0001 |

## Methodology and limitations

The pool is built from reranking candidates, not the full 9.5M passage corpus, so the task is easier than real MIRACL and the absolute values are inflated. MIRACL's hard negatives were chosen by a BM25-style retriever, so in the own-negatives pool every candidate is already a high-BM25-rank document for its query: BM25 can barely separate relevant from non-relevant there, while dense retrieval can, so that pool understates lexical methods (the n-gram trick included) and flatters dense. The n-gram bridge acts when a document is found, not when candidates are reranked, so it can only help in the other-negatives pool, where distractors are topically unrelated and lexical matching discriminates. Use --negatives own|other|mixed to switch. Comparing configurations within one pool is valid; comparing these numbers to published MIRACL results is not.

For Recall@10, MRR@10 and nDCG@10 each non-baseline configuration is compared to bm25-stem pairwise per query: delta is the mean difference, the 95% interval is a paired bootstrap over queries (10000 resamples, fixed seed) and p is a Wilcoxon signed-rank test. There are four comparisons per metric; p_holm applies a Holm-Bonferroni correction across those four and the raw p is kept beside it. The interval reflects only the spread across the queries in this pool, not generalization to other corpora.
