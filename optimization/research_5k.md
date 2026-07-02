# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget 2 WS + tuned-ratio decay epochs. Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

| trained on | ahead | imm | logloss | params | provenance | summary |
|---|---|---|---|---|---|---|
| 101–4999 | — | — | — | 2,762,884 | adopted | Old d=128 leaderboard model, unquantized; the fp target to beat on 5001–10000. |
| 1–5000 | — | — | — | 193,724 | invented | H=2/K=16 champion carried to 5k scale, quant-aware; the phase's starting point. |
