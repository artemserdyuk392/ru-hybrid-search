# MIRACL ru benchmark

Not run yet. Fill this in with `python benchmarks/run.py --subset N`.
The script overwrites this file with real numbers, hardware and timings.

The evaluation pool is the union of the positive and negative passages that
MIRACL ships inline with each dev query (see the module docstring in run.py).
It is a shared-pool benchmark, not full-corpus retrieval over the 9.5M ru
passages, so absolute recall runs high; treat the numbers as relative
comparisons between configurations, not as MIRACL leaderboard scores.

- queries: TODO
- passages in pool: TODO
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

TODO: run on the full dev set (subset omitted) once a GPU box is free; the
CPU e5 pass over the pool is the slow part.
