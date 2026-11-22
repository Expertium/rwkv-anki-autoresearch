"""One-shot: (a) name the fast-rank knobs in the AGENTS.md sentence, (b) add the 14:15 Live-status
bullet to scratchpad/quant_plan/NOTES.md.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

# ---- AGENTS.md: name the two remaining knobs
p = ROOT / "AGENTS.md"
text = p.open(encoding="utf-8", newline="").read()
OLD = ("The flag is OFF by default and must stay OFF for any Python<->Rust parity run: the fast factors "
       "are an approximation, not a bit-identical truncation.")
NEW = ("The knobs are `RWKV_LOWRANK_FAST` (off by default), `RWKV_LOWRANK_FAST_CHECK` (run BOTH solvers, "
       "record the largest relative Frobenius difference, still WRITE the reference factors, so a checked "
       "run is bit-identical to the reference) and `RWKV_LOWRANK_FAST_ITERS` (default 2). The flag must "
       "stay OFF for any Python<->Rust parity run: the fast factors are an approximation, not a "
       "bit-identical truncation.")
if text.count(OLD) != 1:
    print(f"ABORT [AGENTS knobs]: anchor found {text.count(OLD)} times")
    sys.exit(1)
p.open(encoding="utf-8", newline="", mode="w").write(text.replace(OLD, NEW))
print("ok [AGENTS knobs]")

# ---- NOTES.md: the Live-status bullet (newest first)
p = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
text = p.open(encoding="utf-8", newline="").read()
HEAD = "## Live status\n\n"
BULLET = (
    "* 2026-11-22 14:15 - **The rank>=2 NOTE speed blocker is gone: E2 (the fast rank-r solver) is "
    "implemented, proven and measured, and row 29 is now a loss-gate AND speed pass.** "
    "`RWKV_LOWRANK_FAST=1` replaces the KxK Gram + full symmetric eigensolve in `compress_wkv_state` with "
    "a warm-started block subspace iteration on fixed-size arrays + Rayleigh-Ritz on the (R+1)x(R+1) "
    "projection; the tracked block is the R wanted directions plus one extra, refreshed from the strongest "
    "column of A on every call, so a new dominant direction can still enter, and it travels with the "
    "per-entity state (`FastLayerState::basis`). Same session, `it36l2i8n2i8_124.safetensors`, users "
    "107/136/156, 2 reps: `l2i8n2i8` **2,429.2 rev/s = 0.803x** (reference solver 1,801.5 = 0.618x) and "
    "`l3i8n2i8` **2,393.0 = 0.791x** (reference 1,781.4 = 0.611x), against plain 3,025.1 / 2,915.1 rev/s; "
    "the rank-2 CARD configs improve too (2,357.8 -> 2,544.9 = 0.841x). **Proof it is safe to deploy:** "
    "`RWKV_LOWRANK_FAST_CHECK=1` runs BOTH solvers, records the largest relative Frobenius difference and "
    "still WRITES the reference factors, so a checked run is byte-identical to the no-flag run on all "
    "three configs; on `l2i8n2i8` 95.6% of 64,132 rank>=2 calls are under 1e-3 relative, 57 above 1e-1, "
    "none above 1.0, and 3.2% fall back to the reference (degenerate states, cheap there); the scored "
    "deploy cost is imm +0.000051 / ahead +0.000000, the same order as the quantization tax; and a "
    "finiteness run with the flag ON passes (`rust_finite_fastrank.json`: 0 non-finite of 65,671,168 "
    "state values, state_maxabs 0.62). The flag is OFF by default, is recorded in `rust_grid.CONFIGS` for "
    "the three rank>=2-note configs (it does NOT change `bits_for`), and must stay UNSET for any "
    "Python<->Rust parity run. No 0.04-epoch smoke applies: this is an engine solver change with no "
    "training path. **Files:** `rust_fastrank_check.py` + `rust_fastrank_check.json`, "
    "`rust_speed_fastrank_{on,off}.json`, `rust_finite_fastrank.json`, `_analyze_fastrank_drift.py`, the "
    "`compress_bench` test module in `model.rs` (the eigensolve is ~35 us of a ~43 us rank>=2 compress "
    "call). RESULTS.md row 29 + a new fast-solver section; GPU_RUN_BOARD amendment 13. `cargo build "
    "--release` clean, `cargo test --release` 11/11 (the bench is the 12th, ignored). **Also repaired:** "
    "the RESULTS.md paragraph after the card-rank-ladder table was truncated mid-sentence (\"...as the "
    "750-user eval\") with its tail \"does).\" orphaned after the corruption paragraph - the two halves "
    "are rejoined; no text was invented.\n\n")
if text.count(HEAD) != 1:
    print(f"ABORT [NOTES heading]: found {text.count(HEAD)} times")
    sys.exit(1)
p.open(encoding="utf-8", newline="", mode="w").write(text.replace(HEAD, HEAD + BULLET, 1))
print("ok [NOTES bullet]")
