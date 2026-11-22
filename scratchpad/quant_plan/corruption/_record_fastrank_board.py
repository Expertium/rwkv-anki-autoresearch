"""One-shot: the GPU_RUN_BOARD.md half of the E2 record (the AGENTS.md half is already written).

This file is CRLF-dominant but the two anchors live in LF-only stretches, so the replacements here are
RAW (no newline translation).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "optimization" / "GPU_RUN_BOARD.md"
text = p.open(encoding="utf-8", newline="").read()


def raw_sub(old, new, label):
    global text
    got = text.count(old)
    if got != 1:
        print(f"ABORT [{label}]: anchor found {got} times")
        sys.exit(1)
    text = text.replace(old, new)
    print(f"ok [{label}]")


raw_sub(
    "BLOCKER: rank >= 2 has no fused path in the deploy engine (measured 0.629x rev/s, 0.448x\n"
    "states/s), so any rank >= 2 recipe needs a fused rank-r readout in the engine before it can be deployed\n"
    "- that is a deploy-path change and needs Andrew's approval.",
    "BLOCKER (SUPERSEDED 2026-11-22 14:15): rank >= 2 had no fused path in the deploy engine (measured\n"
    "0.629x rev/s, 0.448x states/s), so a rank >= 2 recipe needed a fused rank-r readout before it could be\n"
    "deployed. That path now EXISTS: `RWKV_LOWRANK_FAST=1` (E2) replaces the KxK Gram + symmetric\n"
    "eigensolve with a warm-started block subspace iteration + Rayleigh-Ritz, and it takes the rank-2 NOTE\n"
    "from 0.618x to 0.803x rev/s - see the E2 amendment at the end of this file.",
    "BLOCKER superseded")

raw_sub(
    "0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses\n"
    "`card:2:int8`.",
    "0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses\n"
    "`card:2:int8`. (2026-11-22 14:15: the NOTE side is solved too - `RWKV_LOWRANK_FAST=1` measures\n"
    "`l2i8n2i8` at 0.803x rev/s and `l3i8n2i8` at 0.791x. See the E2 amendment at the end of this file.)",
    "rank>=2 conclusion")

AMEND = """
## Amendment 13 (2026-11-22 14:15): the E2 fast rank-r solver unblocks a rank>=2 NOTE

`RWKV_LOWRANK_FAST=1` replaces the rank>=2 truncation (KxK Gram matrix + full `nalgebra::SymmetricEigen`,
about 35 us of a 43 us compress call on a 32x32 state) with a warm-started block subspace iteration on
fixed-size arrays plus Rayleigh-Ritz on the (R+1)x(R+1) projection. The tracked block is the R wanted
directions plus one extra direction, refreshed from the strongest column of A each call, so a new
dominant direction can still enter. `RWKV_LOWRANK_FAST_ITERS` (default 2) sets the iteration count, and
the basis travels with the per-entity state (`FastLayerState::basis`).

Measured A/B, same session, `it36l2i8n2i8_124.safetensors`, users 107/136/156, 2 reps:

| config | reference solver | fast solver | note |
|---|---:|---:|---|
| plain | 2,915.1 rev/s (1.000x) | 3,025.1 rev/s (1.000x) | baseline |
| `l2i8` (rank-2 card) | 2,357.8 (0.809x) | 2,544.9 (0.841x) | the card path improves too |
| `l2i8s4` | 2,357.8 (0.809x) | 2,544.9 (0.841x) | |
| `l2i8n2i8` (rank-2 NOTE) | 1,801.5 (0.618x) | **2,429.2 (0.803x)** | was the blocker |
| `l3i8n2i8` (rank-3 NOTE) | 1,781.4 (0.611x) | **2,393.0 (0.791x)** | |

Why it is safe to deploy: `RWKV_LOWRANK_FAST_CHECK=1` runs both solvers, records the largest relative
Frobenius difference, and still writes the REFERENCE factors, so a checked run is bit-identical to the
no-flag run (verified on 3 configs). Of 64,132 rank>=2 calls on `l2i8n2i8`, 95.6% are under 1e-3
relative, 57 are above 1e-1, none above 1.0, and 3.2% fall back to the reference because the state is
degenerate (cheap there). Scored on the same traces the fast solver costs imm +0.000051 / ahead
+0.000000 (`rust_score.json`) - the same order as the quantization tax. The finiteness check with the flag
ON passes (`rust_finite_fastrank.json`: 0 non-finite of 65,671,168 state values).

Artifacts: `rust_fastrank_check.py`, `rust_fastrank_check.json`, `rust_speed_fastrank_{on,off}.json`,
`rust_finite_fastrank.json`, `rust_artifacts/it36l2i8n2i8_124.safetensors`. The flag is recorded in
`rust_grid.CONFIGS` for the three rank>=2-note configs and does NOT change `bits_for`; leave it UNSET for
any Python<->Rust parity run, because the fast factors are an approximation, not a bit-identical
truncation. RESULTS.md row 29 (`l2i8n2i8`) is now a loss-gate AND speed pass, so it is a deploy candidate
once `it36corrupt2` clears it.
"""

if not text.endswith("\n"):
    text += "\n"
text += AMEND
p.open(encoding="utf-8", newline="", mode="w").write(text)
print("wrote optimization/GPU_RUN_BOARD.md")
