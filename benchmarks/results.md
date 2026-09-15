# MIRACL ru benchmark (summary)

Three full runs on the Russian MIRACL dev set (997 usable queries), one per
pool mode. Per-mode detail with Recall@10, MRR@10, saturation and the
overlap slice is in results_own.md, results_mixed.md and results_other.md.
Tables below report nDCG@10, the primary metric, and its paired comparison
against bm25-stem (delta, 95% paired-bootstrap CI, Wilcoxon p).

## Difficulty gradient

The three modes differ only in what fills the pool around each query's relevant
passages. own uses the query's own BM25-selected hard negatives (a reranking
task); other uses candidates belonging to other queries (a retrieval task);
mixed takes half of each. Baseline bm25-stem nDCG@10 climbs across them - own
0.347, mixed 0.696, other 0.909 - because own packs the pool with documents
already lexically close to the query, other fills it with topically unrelated
ones, and mixed sits between. own is the hardest and least saturated pool.

## own (99700 documents)

| config | nDCG@10 | delta vs bm25-stem | 95% CI | p |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.3471 |  |  |  |
| bm25-stem+ngram | 0.3302 | -0.0168 | [-0.0339, +0.0001] | 0.0785 |
| dense | 0.6903 | +0.3433 | [+0.3194, +0.3664] | <0.0001 |
| hybrid-stem | 0.5491 | +0.2021 | [+0.1880, +0.2166] | <0.0001 |
| hybrid-stem+ngram | 0.5425 | +0.1954 | [+0.1767, +0.2138] | <0.0001 |

## mixed (20000 documents)

| config | nDCG@10 | delta vs bm25-stem | 95% CI | p |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.6962 |  |  |  |
| bm25-stem+ngram | 0.6510 | -0.0452 | [-0.0627, -0.0276] | <0.0001 |
| dense | 0.8968 | +0.2006 | [+0.1816, +0.2198] | <0.0001 |
| hybrid-stem | 0.8533 | +0.1571 | [+0.1447, +0.1698] | <0.0001 |
| hybrid-stem+ngram | 0.8367 | +0.1405 | [+0.1243, +0.1569] | <0.0001 |

## other (20000 documents)

| config | nDCG@10 | delta vs bm25-stem | 95% CI | p |
| --- | --- | --- | --- | --- |
| bm25-stem | 0.9087 |  |  |  |
| bm25-stem+ngram | 0.8325 | -0.0763 | [-0.0928, -0.0602] | <0.0001 |
| dense | 0.9377 | +0.0290 | [+0.0169, +0.0415] | <0.0001 |
| hybrid-stem | 0.9512 | +0.0425 | [+0.0343, +0.0511] | <0.0001 |
| hybrid-stem+ngram | 0.9279 | +0.0192 | [+0.0073, +0.0310] | 0.0085 |

## The n-gram option: a negative result

The n-gram option was added on a plausible premise. The Russian Snowball
stemmer splits a verb and its deverbal noun into different stems (ускорить ->
ускор, ускорение -> ускорен), so a stem-only index cannot match one against the
other. Character 4-grams of long Russian words share substrings across that
split (уско, скор) and could bridge it.

Across all three pools the effect is negative. bm25-stem+ngram loses to
bm25-stem on nDCG@10 in every mode: -0.0452 in mixed and -0.0763 in other (both
p<0.0001), and -0.0168 in own (p=0.0785, not significant).

The reason is that the gap the trick targets barely exists in MIRACL. Of the 997
queries, 0 have no lexical overlap with their positive documents and 3 have
overlap at or below 0.25; the remaining 994 overlap heavily. There is almost
nothing to bridge, and the cost stays: the n-grams dilute IDF and add spurious
matches, which is what pulls the score down.

Where the trick could still pay off is corpora where queries are written
independently of the documents - search logs, questions against a knowledge
base, support tickets. MIRACL queries were written by annotators who had read
the passage, so query and document nearly always share vocabulary.

## RRF behavior

Unweighted RRF works from ranks alone, so its result depends on how comparable
its inputs are.

- own: dense nDCG@10 0.690 against bm25 0.347; the hybrid lands at 0.549, below
  the better input.
- other: dense 0.938 against bm25 0.909; the hybrid reaches 0.951, above both.

When one component is roughly 1.5 to 2x the other, unweighted RRF drags the
strong side down toward the weak one (own, and mixed at 0.853 below dense 0.897);
only when the two are close does the hybrid beat both (other). Before turning on
the hybrid, measure bm25 and dense separately on your data. If they are that far
apart, use weighted fusion or a reranker instead of unweighted RRF.

## Methodology and limitations

All three pools are built from MIRACL reranking candidates, not the full 9.5M
passage corpus, so every pool document is already a high-BM25-rank document for
some query. That makes the pool harder than a random one for lexical methods and
inflates absolute scores relative to real MIRACL. Recall@100 is at the ceiling
(0.93 to 1.0) and cannot separate configurations; rank by Recall@10, MRR@10 and
nDCG@10. Numbers are comparable between configurations within one mode and are
not comparable to published MIRACL results.

Significance is pairwise per query against bm25-stem: delta is the mean
difference, the 95% interval is a paired bootstrap over queries (10000
resamples, fixed seed), p is a Wilcoxon signed-rank test, and the per-mode files
add a Holm-Bonferroni p_holm across the four comparisons. The intervals reflect
only the spread across these queries, not generalization to other corpora.
