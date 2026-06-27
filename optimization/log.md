# Optimization log (steps 4–5–7)

Regenerated from `log.jsonl` (do not edit by hand). `comment` is in the jsonl only.
Gates: LL not worse than iter0 by >+0.0015 (both modes); state ≤ iter0; size identical.

| # | timestamp | ahead LL | imm LL | params | state KiB | throughput (rev/s) | wilcoxon p | size✓ | LL✓ | state✓ | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-06-27T22:21:25 | 0.374046 | 0.319475 | 2,762,884 | 51.0 | 181.8 | n/a | baseline | baseline | baseline | Frozen baseline: current arch d_model=128, 2.76M params, train 1-100 eval 101-200 |
| 1 | 2026-06-27T22:51:13 | 0.353276 | 0.321249 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter1: halve d_model 128->64 (N_HEADS 4->2), 3.44x fewer params, half state |
| 2 | 2026-06-27T23:12:36 | 0.362885 | 0.326886 | 804,036 | 25.5 | n/a | n/a | PASS | FAIL | PASS | iter2: d_model=64 + IMMEDIATE_SCALE=2.0 to recover imm via ahead-slack |
| 3 | 2026-06-27T23:34:45 | 0.358576 | 0.318373 | 804,036 | 25.5 | 199.3 | n/a | PASS | PASS | PASS | iter3 CHAMPION: d_model=64 + WSD decay phase; both modes beat iter0, 3.44x fewer params |
