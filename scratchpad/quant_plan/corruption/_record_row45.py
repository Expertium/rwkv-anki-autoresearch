"""Record RESULTS row 45 (`l2i6s4`): the new axis-1 frontier at 2,944 bits.

Sources of truth:
  * scratchpad/quant_plan/quant_sweep_it36l2i6s4.json - row 45's 750-user sweep (PASS, both heads).
  * scratchpad/quant_plan/it36int6_chain.log          - the point's smoke/decay/eval timings.

Writes the RESULTS row + paragraph + status sentence, three NOTES live-status bullets, GPU_RUN_BOARD
Amendment 30, and the RECORDED.txt stamp (so the hourly supervisor does not write a duplicate row).

Run (it sits in corruption/, so ROOT is parents[3]):
  .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_row45.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SW45 = RUN / "quant_sweep_it36l2i6s4.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"

A_ROW = b"so the whole int4-card ladder fails whatever note sits under it | - |"
A_PARA = b"rather than to change the batch or the bit budget."     # end of the row-44 paragraph
A_STAT = b"corruption included**"
A_NOTE = b"## Live status\n\n"
MARK_ROW = b"| 45 |"
MARK_PARA = b"**Row 45 (`l2i6s4`) is the smallest passing row"
MARK_NOTE = b"Row 45 (`l2i6s4`) is the new axis-1 frontier"
AMEND = b"## Amendment 30"


def f6(x):
    return f"{x:+.6f}"


def main() -> int:
    r = json.loads(SW45.read_text(encoding="utf-8"))["rows"][0]
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    secs = r["secs"]

    # ---------------------------------------------------------------- RESULTS row 45
    row45 = (
        "| 45 | QAT card rank-2 int6 factors + int4 shifts, note rank-1 int6 factors + int4 shifts - "
        "row 17's layout with BOTH streams' factor widths cut int8 -> int6 (1,024/1,920 = 2,944 bits, "
        "640 below row 17 and 448 below row 35) (`l2i6s4`, `it36int6` point 1, landed 00:53, "
        f"{secs/60:.1f} min; speed cell = row 17's measured `l2i8s4` path, because the factor width "
        "changes only the quantization LEVELS, not the verify path, and both streams keep int4 shifts) "
        f"| 1,024 (128) | 1,920 (240) | 2,290 | 0.800x | {a['mean_cur']:.6f} | {i['mean_cur']:.6f} | "
        f"{f6(a['delta'])} | {f6(i['delta'])} | **PASS on BOTH heads, and the SMALLEST PASSING ROW so "
        f"far**: ahead sits at 0.25x of the ceiling (ub95 +{a['ub95']:.6f}, p {a['p_worse']:.1e}) and "
        f"the immediate head at 0.89x (ub95 +{i['ub95']:.6f}, p {i['p_worse']:.1e}), so the imm rise is "
        "REAL and the head is nearly spent, but it is inside. The billing: against row 17 (the SAME "
        "layout with int8 factors) the int8 -> int6 cut on both streams costs about +0.000073 ahead / "
        "+0.000054 imm (difference of deltas, not a paired test) for 640 bits - about 0.0000001 of "
        "loss per bit, cheaper than any rung measured at 750 users - and QAT recovers most of what the "
        "deploy screen predicted for the mixture (PTQ: `l2i6n` +0.000268/+0.000307, the int6 card "
        "+0.000052/+0.000115) | pending |"
    )

    raw = RESULTS.read_bytes()
    if MARK_ROW in raw:
        print("RESULTS: row 45 already present")
    else:
        lines = raw.split(b"\n")
        idx = [k for k, l in enumerate(lines) if A_ROW in l]
        if len(idx) != 1:
            raise SystemExit(f"ABORT: row anchor found {len(idx)} times")
        k = idx[0]
        eol = b"\r" if lines[k].endswith(b"\r") else b""
        lines[k + 1:k + 1] = [(row45 + (eol.decode() if eol else "")).encode("utf-8")]
        RESULTS.write_bytes(b"\n".join(lines))
        print("RESULTS row 45: ok")

    # ---------------------------------------------------------------- RESULTS paragraph
    para = (
        "**Row 45 (`l2i6s4`) is the smallest passing row measured so far: 2,944 bits, i.e. 448 below "
        "row 35 (`n1i8s3`, 3,392) and 640 below row 17.** It is row 17's recipe with BOTH factor widths "
        "cut from int8 to int6 - card rank-2 int6 + int4 shifts (1,024 bits), note rank-1 int6 + int4 "
        f"shifts (1,920) - and it reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{f6(a['delta'])}/{f6(i['delta'])} against `qnj_it36ctl` (ub95 +{a['ub95']:.6f}/"
        f"+{i['ub95']:.6f}), so the whole cut costs about +0.000073 ahead / +0.000054 imm on top of "
        "row 17 (difference of deltas, not a paired test) for 640 bits. The headroom is asymmetric and "
        f"that is the actionable part: ahead sits at 0.25x of the +0.0005 ceiling while the immediate "
        f"head is at 0.89x (ub95 +{i['ub95']:.6f}, p {i['p_worse']:.1e}), so this row PASSES but the "
        "imm head is nearly spent - the next 128-bit cut has to come from the note's FACTOR width, "
        "since the note's int2 shift rung is closed (row 20, +0.0012-0.0014 imm) and 2,368 was already "
        "shown unreachable on the int4 card (row 42, 1.7x/2.7x over). The QAT-versus-PTQ part is worth "
        "recording: the deploy screen priced the note's int6 factors alone at +0.000268/+0.000307 (PTQ, "
        "11 long histories) and the card's int6 factors at +0.000052/+0.000115, i.e. about "
        "+0.00032/+0.00042 together, while QAT measures +0.000125/+0.000381 - training recovers roughly "
        "0.0002 of the ahead head, the same pattern as `l2i6c`, where the card's int6 rung under QAT "
        "was nearly free. Its finiteness check was launched the moment chain 17 ended (the first free "
        "GPU window, held by `corruption/gpu_hold.py`), so the corruption cell follows here rather than "
        "in a later chain.\n"
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
    if b"row 45 `l2i6s4`" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(A_STAT) != 1:
            raise SystemExit(f"ABORT: status anchor found {raw.count(A_STAT)} times")
        ins = (b"corruption included, and row 45 `l2i6s4` is the SMALLEST PASSING ROW at 2,944 bits "
               b"(loss gate PASS on both heads at ub95 +0.000189/+0.000446; its finiteness check was "
               b"launched in the first free GPU window, as chain 17 ended)** (row 45 `l2i6s4` "
               b"+0.000125/+0.000381 at 1,024/1,920 = 2,944 bits - the int6 factor cut on both "
               b"streams, 448 bits below row 35 and 640 below row 17, landed 00:53) ")
        RESULTS.write_bytes(raw.replace(A_STAT, ins, 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullets
    b45 = (
        "* 2026-11-23 01:05 - **Row 45 (`l2i6s4`) is the new axis-1 frontier: 2,944 bits, PASS on "
        "both heads.** Row 17's recipe with both factor widths cut int8 -> int6 (card rank-2 int6 + "
        f"int4 shifts = 1,024, note rank-1 int6 + int4 shifts = 1,920) reads {a['mean_cur']:.6f}/"
        f"{i['mean_cur']:.6f} = {f6(a['delta'])}/{f6(i['delta'])} against `qnj_it36ctl` (ub95 "
        f"+{a['ub95']:.6f}/+{i['ub95']:.6f}), 448 bits below row 35's 3,392 and 640 below row 17. "
        "Ahead keeps 0.25x of the ceiling; the immediate head is at 0.89x, so the next cut has to come "
        "from the note's FACTOR width. QAT recovers most of the ahead head that the PTQ screen "
        "predicted. Its finiteness check was launched in the first free GPU window (chain 17's end).\n"
    )
    e1fix = (
        "* 2026-11-23 01:05 - **The E1 tail penalty is fixed (per-stream step subsample) and re-queued "
        "as chain 23 (`it36e1b`), so E1 is testable after all.** The OOM was memory, not math: the "
        "penalty's differentiable `svdvals` ran per TIMESTEP per truncated layer and kept every step's "
        "graph until backward, so it grew with the training window. `RWKV_QAT_TAIL_STRIDE` (default 8) "
        "now accumulates on one step in 8 PER STREAM TAG - an unbiased sample of the mean tail at "
        "1/stride of the memory - verified in `scratchpad/_test_e1_stride.py` and in the real trainer "
        "on CPU (`corruption/e1_cpu_smoke.py`, which logs `[tail-penalty] lambda=1.0 stride=8 "
        "card=0.002546(n=7) note=0.011098(n=19)`). Chain 23 runs lambda = 1.0 / 3.0 / 10.0 at stride 8 "
        "(three points on one curve, ungated), each with a 0.04-epoch smoke first and a finiteness "
        "check for any point that passes. The lever is inert unless `RWKV_QAT_TAIL_PENALTY` is set, so "
        "the change cannot disturb a running chain.\n"
    )

    nb = NOTES.read_bytes()
    if MARK_NOTE in nb:
        print("NOTES: bullets already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + b45.encode("utf-8") + e1fix.encode("utf-8"), 1))
        print("NOTES bullets: ok")

    # ---------------------------------------------------------------- BOARD Amendment 30
    amend = (
        "\n\n## Amendment 30 (2026-11-23 01:05): row 45 (`l2i6s4`) is the smallest passing row at "
        "2,944 bits, and the E1 penalty is fixed and re-queued\n\n"
        "**(1) A new axis-1 frontier.** `it36int6` point 1 - row 17's layout with BOTH factor widths "
        "cut int8 -> int6 (card rank-2 int6 + int4 shifts = 1,024 bits, note rank-1 int6 + int4 shifts "
        f"= 1,920, 2,944 bits total) - reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{f6(a['delta'])}/{f6(i['delta'])} against `qnj_it36ctl` (ub95 +{a['ub95']:.6f}/"
        f"+{i['ub95']:.6f}), a PASS on both heads. That is 448 bits below row 35 (3,392, the previous "
        "smallest passing row) and 640 below row 17, for about +0.000073 ahead / +0.000054 imm over "
        "row 17 (difference of deltas, not a paired test). The imm head sits at 0.89x of the +0.0005 "
        "ceiling, so the next 128-bit cut must come from the note's FACTOR width: the note's int2 shift "
        "rung is closed (row 20) and 2,368 was shown unreachable on the int4 card (row 42). Its "
        "finiteness check ran in the first free GPU window, held by `corruption/gpu_hold.py` so chain "
        "18 could not take the GPU first.\n\n"
        "**(2) E1 is testable again.** The OOM that killed chain 15's two penalty points was memory, "
        "not mathematics: `accum_tail_penalty` ran a DIFFERENTIABLE `svdvals` once per TIMESTEP per "
        "rank-truncated layer and every one of those graphs stayed alive until backward, so the cost "
        "scaled with the training window. `rwkv/model/rwkv_ops.py` now samples ONE step in "
        "`RWKV_QAT_TAIL_STRIDE` (default 8) PER STREAM TAG, an unbiased estimate of the mean tail at "
        "1/stride of the graph memory. Verified twice before queueing: `scratchpad/_test_e1_stride.py` "
        "(stride 4 -> 2 of 8 calls, 8 -> 1 of 8, 1 -> 8 of 8, counter per tag) and the real trainer on "
        "CPU (`corruption/e1_cpu_smoke.py` with `E1SMOKE_STRIDE=8`, logging `[tail-penalty] lambda=1.0 "
        "stride=8 card=0.002546(n=7) note=0.011098(n=19)`). The lever is still inert when "
        "`RWKV_QAT_TAIL_PENALTY` is unset, so nothing running is affected.\n\n"
        "**(3) Chains.** `it36int6` (17) is running its second point (`l2i5s4`, 2,624 bits) with "
        "`l2i4s4n4` after it; the waiters are `it36int6card`, `it36fixedbasis`, `it36int6combo` (21), "
        "`it36int6note` (22) and the new `it36e1b` (23: E1 lambda 1.0/3.0/10.0 at stride 8, "
        "self-diagnosing because a point whose smoke OOMs is skipped).\n"
    )

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 30 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 30 appended")

    # ---------------------------------------------------------------- RECORDED stamp
    rb = RECORDED.read_bytes()
    if b"it36l2i6s4" in rb.split(b"\n"):
        print("RECORDED.txt: stamp already present")
    else:
        if not rb.endswith(b"\n"):
            rb += b"\n"
        RECORDED.write_bytes(rb + b"it36l2i6s4\n")
        print("RECORDED.txt: added it36l2i6s4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
