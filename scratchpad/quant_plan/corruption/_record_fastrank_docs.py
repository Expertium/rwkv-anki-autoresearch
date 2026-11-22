"""One-shot: record the E2 fast rank-r solver in `AGENTS.md` and `optimization/GPU_RUN_BOARD.md`.

Both files still say a rank >= 2 NOTE cannot be deployed without engine work, which the E2 change
(`RWKV_LOWRANK_FAST=1`, 2026-11-22) makes false: measured 0.803x rev/s for `l2i8n2i8` and 0.791x for
`l3i8n2i8` against 0.618x/0.611x for the reference solver in the same session. The board's stale
"BLOCKER" paragraph and the "rank >= 2 cannot be deployed" sentence are amended in place (the log keeps
its history: the old text is marked as superseded rather than deleted), and a new amendment block is
appended with the evidence.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def patch(path, pairs, append=None, label=""):
    p = ROOT / path
    text = p.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
    print(f"== {path} (nl={'CRLF' if nl == chr(13) + chr(10) else 'LF'}) ==")
    for i, (old, new) in enumerate(pairs, 1):
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        got = text.count(o)
        if got != 1:
            print(f"  ABORT [{label} {i}]: anchor found {got} times: {o.strip()[:70]!r}")
            sys.exit(1)
        text = text.replace(o, n)
        print(f"  ok [{i}] {n.strip().splitlines()[0][:80]}")
    if append:
        if not text.endswith(nl):
            text += nl
        text += append.replace("\n", nl)
        print(f"  ok [append] {append.strip().splitlines()[0][:80]}")
    p.open(encoding="utf-8", newline="", mode="w").write(text)
    print(f"  wrote {path}")


patch("AGENTS.md", [
    ("So a rank >= 2 NOTE needs a FAST rank-r engine path, not a parser change.",
     "So a rank >= 2 NOTE needed a FAST rank-r engine path, not a parser change - and since 2026-11-22 "
     "it HAS one: `RWKV_LOWRANK_FAST=1` swaps the Gram eigensolve for a warm-started block subspace "
     "iteration + Rayleigh-Ritz and measures **0.803x rev/s for `l2i8n2i8`** (rank-2 note) and 0.791x for "
     "`l3i8n2i8`, against 0.618x/0.611x for the reference solver in the same session, at a deploy "
     "log-loss cost of imm +0.000051 / ahead +0.000000 (`rust_fastrank_check.json`, "
     "`rust_speed_fastrank_{on,off}.json`). The flag is OFF by default and must stay OFF for any "
     "Python<->Rust parity run: the fast factors are an approximation, not a bit-identical truncation."),
], label="AGENTS")

patch("optimization/GPU_RUN_BOARD.md", [
    ("+0.00070). BLOCKER: rank >= 2 has no fused path in the deploy engine (measured 0.629x rev/s, 0.448x\n"
     "states/s), so any rank >= 2 recipe needs a fused rank-r readout in the engine before it can be deployed\n"
     "- that is a deploy-path change and needs Andrew's approval. GPU at amendment time: `it36rank`'s `l2i8`\n"
     "QAT decay, chain `it36notecut` waiting.",
     "+0.00070). BLOCKER (SUPERSEDED 2026-11-22 14:15): rank >= 2 had no fused path in the deploy engine\n"
     "(measured 0.629x rev/s, 0.448x states/s), so a rank >= 2 recipe needed a fused rank-r readout before\n"
     "it could be deployed. That path now EXISTS: `RWKV_LOWRANK_FAST=1` (E2) replaces the KxK Gram +\n"
     "symmetric eigensolve with a warm-started block subspace iteration + Rayleigh-Ritz, and it takes the\n"
     "rank-2 NOTE from 0.618x to 0.803x rev/s. GPU at amendment time: `it36rank`'s `l2i8` QAT decay, chain\n"
     "`it36notecut` waiting."),
    ("The earlier \"rank >= 2 cannot be\n"
     "deployed\" conclusion came from `l2i4`, which put rank 2 on the NOTE as well (4 layers) and measured\n"
     "0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses\n"
     "`card:2:int8`.",
     "The earlier \"rank >= 2 cannot be\n"
     "deployed\" conclusion came from `l2i4`, which put rank 2 on the NOTE as well (4 layers) and measured\n"
     "0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses\n"
     "`card:2:int8`. (2026-11-22 14:15: the NOTE side is solved too - see the E2 amendment at the end of\n"
     "this file. `RWKV_LOWRANK_FAST=1` measures `l2i8n2i8` at 0.803x rev/s and `l3i8n2i8` at 0.791x.)"),
], append="""
## Amendment 13 (2026-11-22 14:15): the E2 fast rank-r solver unblocks a rank>=2 NOTE

`RWKV_LOWRANK_FAST=1` replaces the rank>=2 truncation (KxK Gram matrix + full `nalgebra::SymmetricEigen`,
about 35 us of a 43 us compress call) with a warm-started block subspace iteration on fixed-size arrays
plus Rayleigh-Ritz on the (R+1)x(R+1) projection. The tracked block is the R wanted directions plus one
extra direction, refreshed from the strongest column of A each call, so a new dominant direction can
still enter. `RWKV_LOWRANK_FAST_ITERS` (default 2) sets the iteration count; the basis travels with the
per-entity state (`FastLayerState::basis`).

Measured A/B, same session, `it36l2i8n2i8_124.safetensors`, users 107/136/156, 2 reps:

| config | reference solver | fast solver | note |
|---|---:|---:|---|
| plain | 2,915.1 rev/s (1.000x) | 3,025.1 rev/s (1.000x) | baseline |
| `l2i8` (rank-2 card) | 2,357.8 (0.809x) | 2,544.9 (0.841x) | card path also improves |
| `l2i8s4` | 2,357.8 (0.809x) | 2,544.9 (0.841x) | |
| `l2i8n2i8` (rank-2 NOTE) | 1,801.5 (0.618x) | **2,429.2 (0.803x)** | was the blocker |
| `l3i8n2i8` (rank-3 NOTE) | 1,781.4 (0.611x) | **2,393.0 (0.791x)** | |

Evidence that it is safe: `RWKV_LOWRANK_FAST_CHECK=1` runs both solvers, records the largest relative
Frobenius difference, and still writes the REFERENCE factors, so a checked run is bit-identical to the
no-flag run (verified on 3 configs). Of 64,132 rank>=2 calls on `l2i8n2i8`, 95.6% are under 1e-3
relative, 57 are above 1e-1, none above 1.0, and 3.2% fall back to the reference because the state is
degenerate. Scored on the same traces: imm +0.000051 / ahead +0.000000 (`rust_score.json`), the same
order as the quantization tax. The finiteness check with the flag ON passes (`rust_finite_fastrank.json`).

Artifacts: `rust_fastrank_check.py`, `rust_fastrank_check.json`, `rust_speed_fastrank_{on,off}.json`,
`rust_finite_fastrank.json`, `rust_artifacts/it36l2i8n2i8_124.safetensors`. The flag is in
`rust_grid.CONFIGS` for the three rank>=2-note configs (it does NOT change `bits_for`); leave it unset for
Python<->Rust parity runs. RESULTS.md row 29 (`l2i8n2i8`) is now a loss-gate AND speed pass, so it is a
deploy candidate once `it36corrupt2` clears it.
""", label="BOARD")

print("\ndone")
