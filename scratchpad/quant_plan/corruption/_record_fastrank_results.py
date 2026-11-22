"""One-shot: record the E2 fast rank-r solver in `scratchpad/quant_plan/RESULTS.md`.

Four edits:
  1. row 29 (`l2i8n2i8`) gains its measured speed cells (2,429 rev/s = 0.803x) and a verdict that no
     longer calls speed the blocker;
  2. the row-29 footnote sentence that said "a rank >= 2 note needs a FAST rank-r engine path" now says
     it has one;
  3. a new section with the fast solver's evidence (A/B speed, check-mode bit-identity, the difference
     histogram, the scored log-loss cost, the finiteness run);
  4. a REPAIR: the "no engine change is needed ON THE CARD" paragraph was truncated mid-sentence
     ("...as the 750-user eval") and its tail "does)." was left orphaned after the corruption paragraph.
     The two halves are rejoined; no text is invented.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
text = p.open(encoding="utf-8", newline="").read()
nl = "\r\n" if text.count("\r\n") >= text.count("\n") - text.count("\r\n") else "\n"
print(f"nl={'CRLF' if nl == chr(13) + chr(10) else 'LF'}")


def sub(old, new, label, count=1):
    global text
    o, n = old.replace("\n", nl), new.replace("\n", nl)
    got = text.count(o)
    if got != count:
        print(f"ABORT [{label}]: anchor found {got} times, expected {count}")
        sys.exit(1)
    text = text.replace(o, n)
    print(f"ok [{label}]")


# 1. row 29 speed cells
sub("| 3,840 (480) | *see note* | *see note* | 0.333687",
    "| 3,840 (480) | 2,429 | 0.803x | 0.333687", "row 29 speed cells")

# 2. row 29 verdict
sub("the blocker is SPEED - the engine's rank-2 NOTE path reads 0.600x, under the 0.75x floor | pending |",
    "the engine's rank-2 NOTE path reads 0.600x on the REFERENCE solver but **0.803x with the fast rank-r "
    "solver** (`RWKV_LOWRANK_FAST=1`, states/s 0.688x, median per-user 0.810x, users 107/136/156, "
    "`rust_speed_fastrank_{on,off}.json`), so it clears the 0.75x floor; the fast solver's own deploy cost "
    "is imm +0.000051 / ahead +0.000000 on the same traces (`rust_fastrank_check.json`, `rust_score.json`) "
    "| pending |", "row 29 verdict")

# 3. the footnote sentence about the missing path
sub("so a rank >= 2 note needs a FAST rank-r engine path, not a parser change.",
    "so a rank >= 2 note needed a FAST rank-r engine path, not a parser change - **it has one since "
    "2026-11-22 (E2, `RWKV_LOWRANK_FAST=1`); see the fast-solver section below**.",
    "footnote: the path exists")

# 4. the footnote's row-29 sentence
sub("Row 29 (`l2i8n2i8`) is the first row with a rank-2 NOTE that passes the loss gate, and it runs on that "
    "same 0.600x note path (its card is row 17's `l2i8s4` at 0.800x), so it is a loss-gate pass but NOT a "
    "deployable config until the engine gets a fast rank-r note path.",
    "Row 29 (`l2i8n2i8`) is the first row with a rank-2 NOTE that passes the loss gate, and with the fast "
    "solver it clears the speed floor as well, so it is a deploy candidate once its corruption check passes.",
    "footnote: row 29 deployable")

# 5. the new section
SECTION = """**The rank>=2 NOTE speed blocker is GONE: the fast rank-r solver (E2, 2026-11-22).**
`compress_wkv_state` solved rank >= 2 with a KxK Gram matrix plus a full symmetric eigendecomposition per
layer per step (~35 us of a ~43 us compress call on a 32x32 state, measured by the `compress_bench` test
in `model.rs`), while rank 1 used a power iteration - which is why a rank-2 NOTE read 0.618x rev/s and
rank 1 read 0.902x. `RWKV_LOWRANK_FAST=1` replaces the eigensolve with a warm-started block subspace
iteration on fixed-size arrays (never forming the Gram) plus a Rayleigh-Ritz step on the (R+1)x(R+1)
projection; the tracked block is the R wanted directions plus ONE extra direction, refreshed from the
strongest column of A on every call, so a NEW dominant direction can still enter. Same session, same
weights (`it36l2i8n2i8_124.safetensors`), same users (107/136/156), 2 reps: `l2i8n2i8` **2,429.2 rev/s =
0.803x** (reference solver 1,801.5 = 0.618x) and `l3i8n2i8` **2,393.0 = 0.791x** (reference 1,781.4 =
0.611x), against plain 3,025.1 / 2,915.1 rev/s; the rank-2 CARD configs improve too, 2,357.8 -> 2,544.9
rev/s (0.809x -> 0.841x). Files: `rust_speed_fastrank_on.json`, `rust_speed_fastrank_off.json`.

**The fast solver's own cost is measured, not assumed.** `RWKV_LOWRANK_FAST_CHECK=1` runs BOTH solvers,
records the largest relative Frobenius difference between the two rank-r reconstructions, and still
WRITES the reference factors - so a checked run reproduces the reference run bit for bit: its
`rust_pred_*.json` dumps are byte-identical to the no-flag run on all three configs
(`rust_fastrank_check.json`). The difference itself is concentrated: of 64,132 rank>=2 calls on
`l2i8n2i8`, 61,339 (95.6%) are under 1e-3 relative, 256 under 1e-2, 415 under 1e-1, 57 above 1e-1 and
NONE above 1.0, while 2,065 calls (3.2%) fall back to the reference because the state is degenerate
(`mgs_into` rejects a rank-deficient block), which is cheap on such a state. The largest single-review
p(recall) change is 8.1e-2 (p99 7.1e-3, p999 4.3e-2 over 16,033 immediate-head predictions), so the
MAX is a few outlier reviews rather than a systematic shift. The deploy-relevant number is the scored log
loss on those same traces: `RWKV_LOWRANK_FAST=1` costs **imm +0.000051 / ahead +0.000000** on `l2i8n2i8`
(`l3i8n2i8` +0.000225 / -0.000012; `l2i8s4` -0.000031 / -0.000004), the same order as the quantization
tax it sits next to. It is a DEPLOY-side solver choice only - the 750-user gate numbers come from the
torch eval path, which uses the reference truncation - and `bits_for` ignores the flag, so no bit budget
moves. A finiteness run with the flag on still PASSES (`rust_finite_fastrank.json`: 0 non-finite of
65,671,168 card/note state observations and 29,347 p(recall) values, state_maxabs 0.62). Python<->Rust
PARITY runs must NOT set the flag: the fast factors are not bit-identical to the reference by construction.

"""
sub("**The `corruption` column (added 2026-11-21, Andrew's gate).**",
    SECTION + "**The `corruption` column (added 2026-11-21, Andrew's gate).**", "new E2 section")

# 6. the repair (RAW strings: this paragraph boundary mixes a bare LF with the file's CRLF)
def raw_sub(old, new, label):
    global text
    got = text.count(old)
    if got != 1:
        print(f"ABORT [{label}]: anchor found {got} times, expected 1")
        sys.exit(1)
    text = text.replace(old, new)
    print(f"ok [{label}]")


raw_sub("750-user eval\r\n\n",
        "750-user eval does).\r\n\r\n",
        "repair: rejoin the sentence")
raw_sub("no cell is `pending` now.\r\ndoes).\r\n",
        "no cell is `pending` now.\r\n",
        "repair: drop the orphaned tail")

p.open(encoding="utf-8", newline="", mode="w").write(text)
print("wrote RESULTS.md")
