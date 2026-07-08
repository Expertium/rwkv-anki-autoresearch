# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget 2 WS + tuned-ratio decay epochs. Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

`p-value` = paired per-user one-sided Wilcoxon (candidate vs the then-current champion, same 5000 eval
users; `optimization/paired_pvalue.py`), shown `ahead / imm`. **Accept gate (Andrew 2026-07-08): BOTH
modes need p < 0.0001** in addition to the ≥0.0003-both-modes improvement.

| trained on | ahead | imm | logloss | p-value | params | provenance | summary |
|---|---|---|---|---|---|---|---|
| 101–4999 | 0.2964 | 0.2649 | exact | — (reference) | 2,762,884 | adopted | Old d=128 leaderboard model, unquantized; the fp target to beat on 5001–10000. Evaluated 2026-07-03 (n=5000 both modes, full precision: ahead 0.296385 / imm 0.264905). |
| 1–5000 | 0.3066 | 0.2783 | exact | 1.0 / 1.0 (vs target) | 193,724 | invented | **champ5k_r1 = the 5k CHAMPION (starting point).** H=2/K=16, quant-aware q72u with per-run learnable cbs, champion HPs, 2ep WS + 0.5ep decay. Behind the d=128 fp target by 0.0102/0.0134 (the gap the phase closes). Promoted 2026-07-08 (ckpt+cbs in champion_5k.json; WS trace = prune ref). |
