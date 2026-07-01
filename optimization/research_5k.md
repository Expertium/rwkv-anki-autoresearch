# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget 2 WS + 0.5 decay epochs. Detail, reasoning, and status → [research_5k_notes.md](research_5k_notes.md).

| model | trained on | eval 5001–10000 · ahead | imm | params | status |
|---|---|---|---|---|---|
| d=128 baseline (`RWKV_trained_on_101_4999`) | 101–4999 | — | — | 2.76M | to beat — pending eval data |
| H=2/K=16 (ours) | 1–5000 | — | — | 194k | pending — data prep deferred |
