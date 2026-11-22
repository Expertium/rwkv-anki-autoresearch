"""Record row 44 (`l2i8s4pf`, the E1 control) and the E1-penalty OOM failure.

Sources of truth:
  * scratchpad/quant_plan/quant_sweep_it36l2i8s4pf.json  - row 44's 750-user sweep.
  * scratchpad/quant_plan/it36e1_chain.log               - both penalty points OOM-failed.
  * scratchpad/quant_plan/qat_it36l2i8s4e1{a,b}smoke/train.log - the OOM tracebacks.

Writes a RESULTS row + paragraph, two NOTES live-status bullets, GPU_RUN_BOARD Amendment 29 and the
RECORDED.txt stamp.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW44 = RUN / "quant_sweep_it36l2i8s4pf.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

A_ROW = b"so the whole int4-card ladder fails whatever note sits under it | - |"
A_PARA = (b"this screen is PTQ against weights trained for the rank-1-int8 note, so it cannot see what "
          b"QAT recovers.")
A_STAT = b"corruption included**"
A_NOTE = b"## Live status\n\n"
MARK_ROW = b"| 44 |"
MARK_PARA = b"**Row 44 (`l2i8s4pf`) is the E1 control"
MARK_NOTE = b"Row 44 (`l2i8s4pf`, the E1 control) PASSES"
AMEND = b"## Amendment 29"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW44.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    secs = r["secs"]

    # ---------------------------------------------------------------- RESULTS row 44
    row44 = (
        "| 44 | QAT card rank-2 int8 + int4 shifts, note rank-1 int8 + int4 shifts - row 17's EXACT "
        "layout (1,280/2,304 = 3,584 bits) trained on the **Python per-step QAT loop** "
        "(`RWKV_QAT_FUSED=0`) with NO penalty, the E1 control (`l2i8s4pf`, `it36e1` point 1, landed "
        f"00:17, {secs/60:.1f} min; speed cell = row 17's measured `l2i8s4` path, because the bit "
        "layout is identical) | 1,280 (160) | 2,304 (288) | 2,290 | 0.800x | "
        f"{a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {f6(a['delta'])} | {f6(i['delta'])} | "
        f"**PASS**: ub95 +{a['ub95']:.6f}/+{i['ub95']:.6f}. The point of the row is the PARITY it "
        "establishes: the fused-off Python twin reproduces row 17 (fused kernels) to about 2e-6 "
        "(+0.000052/+0.000327 against +0.000050/+0.000325), so the E1 penalty's own delta can be read "
        "straight off row 17 and no part of it is the fused/Python difference | pending |"
    )

    raw = RESULTS.read_bytes()
    if MARK_ROW in raw:
        print("RESULTS: row 44 already present")
    else:
        lines = raw.split(b"\n")
        idx = [k for k, l in enumerate(lines) if A_ROW in l]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times")
        k = idx[0]
        eol = b"\r" if lines[k].endswith(b"\r") else b""
        lines[k + 1:k + 1] = [(row44 + (eol.decode() if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 44: ok")

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        "**Row 44 (`l2i8s4pf`) is the E1 control: the fused-off Python QAT path reproduces the fused "
        "kernels, and the tail-energy penalty itself cannot be trained at the standard batch.** Two "
        "results from `it36e1`. (1) The control - row 17's exact layout (1,280/2,304 = 3,584 bits) "
        f"trained on the Python per-step QAT loop with the penalty OFF - reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}), i.e. row 17's own +0.000052/+0.000327 to about 2e-6. So "
        "`RWKV_QAT_FUSED=0` is bit-equivalent to the fused kernels at the gate's resolution, which is "
        "what makes E1 measurable: the penalty requires the Python loop (the fused rank-1 kernel "
        "truncates inside the kernel, where the pre-truncation state is invisible), so without this "
        "control a small E1 delta could have been the fused/Python difference rather than the penalty. "
        "(2) BOTH penalty points then failed before the first training step: `l2i8s4e1a` (lambda=1.0) "
        "and `l2i8s4e1b` (lambda=10.0) each raised `torch.OutOfMemoryError` at 10.34-10.44 GiB of the "
        "11.00 GiB cap inside `RWKV7_WKV`'s Python loop, 14 s and 27 s in, so the chain skipped them "
        "and logged its end marker at 00:19:35. The penalty as implemented calls a DIFFERENTIABLE "
        "`torch.linalg.svdvals` on the state at EVERY timestep of every truncated layer and keeps each "
        "step's SVD graph alive until backward (`rwkv/model/rwkv_ops.py accum_tail_penalty`), which "
        "adds about 3x the state-graph memory per step on top of the control's own usage; the control "
        "fits the cap and the penalty does not. E1 is therefore NOT refuted - it is untested, and the "
        "fix is to make the penalty cheap (compute it on a subsample of steps, or take it from the "
        "already-computed rank-r reconstruction as a Frobenius residual) rather than to change the "
        "batch or the bit budget.\n"
    )

    raw = RESULTS.read_bytes()
    if MARK_PARA in raw:
        print("RESULTS: paragraph already present")
    else:
        if raw.count(A_PARA) != 1:
            raise SystemExit(f"ABORT: paragraph anchor found {raw.count(A_PARA)} times")
        RESULTS.write_bytes(raw.replace(A_PARA, A_PARA + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS paragraph: ok")

    # ---------------------------------------------------------------- status sentence
    raw = RESULTS.read_bytes()
    if b"row 44 `l2i8s4pf`" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(A_STAT) != 1:
            raise SystemExit(f"ABORT: status anchor found {raw.count(A_STAT)} times")
        ins = (b"corruption included** (row 44 `l2i8s4pf` +0.000050/+0.000325 - the E1 control, row "
               b"17's layout on the Python QAT loop, reproduces the fused kernels to 2e-6; the two "
               b"penalty points then OOM'd, landed 00:17)")
        RESULTS.write_bytes(raw.replace(A_STAT, ins, 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullets
    b44 = (
        "* 2026-11-23 00:20 - **Row 44 (`l2i8s4pf`, the E1 control) PASSES and proves the fused-off "
        "Python QAT loop matches the fused kernels to 2e-6.** Row 17's exact layout "
        f"(1,280/2,304 = 3,584 bits) on `RWKV_QAT_FUSED=0` with no penalty reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}) against `qnj_it36ctl`, i.e. row 17's +0.000052/+0.000327 "
        "to about 2e-6. That parity is what makes the E1 lever measurable: the tail penalty needs the "
        "Python loop (the fused rank-1 kernel truncates inside the kernel), so its delta can now be "
        "read straight off row 17.\n"
    )
    boom = (
        "* 2026-11-23 00:20 - **The E1 tail penalty cannot be trained at the standard batch: both "
        "penalty smokes OOM'd, so E1 is UNTESTED, not refuted.** `it36e1` ran `l2i8s4e1a` (lambda=1.0) "
        "and `l2i8s4e1b` (lambda=10.0) and each raised `torch.OutOfMemoryError` at 10.34-10.44 GiB of "
        "the 11.00 GiB cap inside `RWKV7_WKV`'s Python loop, 14 s and 27 s in - before the first "
        "training step - so the chain skipped both and logged `=== it36e1 chain end ===` at 00:19:35; "
        "`it36e1shrink` then read both gate inputs as `null` and ran nothing. Cause: "
        "`accum_tail_penalty` (`rwkv/model/rwkv_ops.py`) calls a DIFFERENTIABLE "
        "`torch.linalg.svdvals` on the state at EVERY timestep of every truncated layer and keeps each "
        "step's SVD graph alive until backward, adding roughly 3x the state-graph memory per step over "
        "the control. The control (`l2i8s4pf`) fits the cap; the penalty does not. The fix is to make "
        "the penalty cheap (a step subsample, or a Frobenius residual off the already-computed rank-r "
        "reconstruction), NOT to change the batch size or the bit budget.\n"
    )

    nb = NOTES.read_bytes()
    if MARK_NOTE in nb:
        print("NOTES: bullets already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + b44.encode("utf-8") + boom.encode("utf-8"), 1))
        print("NOTES bullets: ok")

    # ---------------------------------------------------------------- BOARD Amendment 29
    amend = (
        "\n\n## Amendment 29 (2026-11-23 00:20): the E1 control PASSES and the Python QAT loop matches "
        "the fused kernels; the E1 penalty itself OOMs at the standard batch, so E1 is untested\n\n"
        "**(1) The E1 control is clean.** `l2i8s4pf` - row 17's exact layout (card rank-2 int8 + int4 "
        "shifts, note rank-1 int8 + int4 shifts, 1,280/2,304 = 3,584 bits) trained on the Python "
        "per-step QAT loop (`RWKV_QAT_FUSED=0`) with NO penalty - reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}) against `qnj_it36ctl`. Row 17 (fused kernels) is "
        "+0.000052/+0.000327, so the two paths agree to about 2e-6 and `RWKV_QAT_FUSED=0` is "
        "bit-equivalent to the fused kernels at the gate's resolution. This is the control the E1 "
        "design needs: the penalty must run on the Python loop, so its delta can now be attributed to "
        "the penalty rather than to the fused/Python difference. 37.4 min for the 750-user sweep.\n\n"
        "**(2) The penalty OOMs, so E1 has no verdict.** `it36e1` then ran `l2i8s4e1a` (lambda=1.0) and "
        "`l2i8s4e1b` (lambda=10.0); both raised `torch.OutOfMemoryError` at 10.34-10.44 GiB of the "
        "11.00 GiB allocator cap inside `RWKV7_WKV`'s Python loop, after 14 s and 27 s and BEFORE the "
        "first training step. The chain skipped both points and logged its end marker at 00:19:35; "
        "`it36e1shrink` read `it36l2i8s4e1a`/`it36l2i8s4e1b` as `null`, found no chain-15 penalty "
        "sweep, and ran nothing. The mechanism is the penalty's own cost: `accum_tail_penalty` calls a "
        "DIFFERENTIABLE `torch.linalg.svdvals` on the state at every timestep of every rank-truncated "
        "layer and keeps each step's SVD graph alive until backward, which adds roughly 3x the "
        "state-graph memory per step over the control. The control fits the cap; the penalty does not. "
        "E1 is therefore UNTESTED rather than refuted, and the cheap fix is to subsample the steps the "
        "penalty sees (or take the tail off the rank-r reconstruction already computed in "
        "`fake_lowrank_state` as a Frobenius residual) - not to shrink the batch or the bit budget.\n\n"
        "**(3) Chain state.** `it36e1` (chain 15) and `it36e1shrink` (chain 16) both ENDED at "
        "00:19:35-00:19:38; `it36int6` (chain 17) took the GPU at 00:19:59 with `l2i6s4`, whose smoke "
        "passed in 1.2 min, and its 124-step decay is running. The waiters behind it are "
        "`it36int6card`, `it36fixedbasis`, `it36int6combo` (chain 21) and `it36int6note` (chain 22).\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 29 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 29 appended")

    # ---------------------------------------------------------------- RECORDED stamp
    rb = RECORDED.read_bytes()
    if b"it36l2i8s4pf" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l2i8s4pf\n")
        print("RECORDED.txt: added it36l2i8s4pf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
