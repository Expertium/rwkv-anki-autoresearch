# RWKV 5k phase

Train **1–5000**, eval **5001–10000** (held-out half); budget 2 WS + tuned-ratio decay epochs. Detail & reasoning → [research_5k_notes.md](research_5k_notes.md).

| trained on | ahead | imm | logloss | params | provenance | summary |
|---|---|---|---|---|---|---|
| 101–4999 | 0.2964 | 0.2649 | exact | 2,762,884 | adopted | Old d=128 leaderboard model, unquantized; the fp target to beat on 5001–10000. Evaluated 2026-07-03 (n=5000 both modes, full precision: ahead 0.296385 / imm 0.264905). |
| 1–5000 | — | — | — | 193,724 | invented | H=2/K=16 champion carried to 5k scale, quant-aware; the phase's starting point. |
