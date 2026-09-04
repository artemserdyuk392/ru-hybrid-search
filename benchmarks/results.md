# MIRACL ru benchmark

Not run yet. Fill this in with `python benchmarks/run.py` (defaults: 100
queries, 20000 documents). The script overwrites this file with real numbers,
the exact sample, hardware and timings.

The corpus is every qrels document of the evaluated queries plus random
distractor passages drawn from the dev split, up to --subset documents. It is
a shared-pool benchmark over the dev passages, not full-corpus retrieval over
the 9.5M ru passages, so treat the numbers as relative comparisons between
configurations, not as MIRACL leaderboard scores.

- queries: TODO
- documents: TODO
- relevant documents: TODO
- rng seed: TODO
- candidates per method: 200
- hardware: TODO
- python: TODO

| config | Recall@100 | nDCG@10 |
| --- | --- | --- |
| bm25-stem | - | - |
| bm25-stem+ngram | - | - |
| dense | - | - |
| hybrid-stem | - | - |
| hybrid-stem+ngram | - | - |

TODO: run on more queries once a GPU box is free; the CPU e5 pass over 20000
passages is the slow part.
