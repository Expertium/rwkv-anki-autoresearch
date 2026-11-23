"""Chain 25: the SHIFT-width rungs below row 45 -> 2,880 and 2,688 bits, then their corruption checks.

WHY THIS CHAIN EXISTS AGAIN. The `it36wide7` screen priced these three rungs against row 45's own
`l2i6s4` and every one of them was REJECTED on the immediate head, with the recorded reason (NOTES.md
2026-11-23 02:45, GPU_RUN_BOARD Amendment 33):

    "Row 45's immediate head already uses 0.89x of the +0.0005 ceiling (ub95 +0.000446), so a rung that
     costs +0.00046 imm cannot pass unless QAT recovered ~90% of it - which it does not (the note's int6
     cut recovered 9% of its PTQ imm price)."

That parenthetical inverts its own number, and chain 18's 750-user runs now measure the quantity directly.
The note's int6 cut (row 49 minus row 17) costs +0.000046 ahead / +0.000031 imm after QAT, against the
11-user screen's PTQ price of +0.000268/+0.000307 for the same cut (`it36int6split`). So the QAT cost is
about 10-17% OF the screen's price - i.e. QAT removes 83-90% of it, which is the very condition the
sentence above says would let a shift rung PASS. The same comparison for `l2i6s4` as a whole (screen
+0.000442/+0.000726 on 21 users vs QAT +0.000125/+0.000381) gives 28%/53%, and for row 35's note shift cut
the screen and QAT agree closely (-0.000036 vs +0.000037 imm). So a PTQ deploy screen does NOT predict the
QAT cost of a cut; the fraction it removes varies from ~10% to ~90% across the cuts measured so far, and
the shift rungs were closed on the assumption that it could not exceed ~10%.

WHAT IS THEREFORE STILL LIVE. Row 45 (2,944 bits) is the frontier, and these two rungs are the only
pre-registered candidates below it other than chain 22's 2,752:

    config           card/note bits   d ahead (7 users, PTQ)        d imm (7 users, PTQ)
    l2i6s3n1i6s4      960 / 1,920      +0.000071 [-0.000082,+0.000245]  +0.000497 [+0.000044,+0.001095]
    l2i6s3n1i6s3      960 / 1,728      +0.000139 [-0.000049,+0.000317]  +0.000461 [+0.000031,+0.000993]

The ahead head is nearly free on both (its interval CONTAINS zero), so the whole question is whether the
immediate head's PTQ price survives QAT. Row 45's remaining imm headroom is +0.000119 (ub95 +0.000446 of
+0.0005), so a rung passes if QAT removes more than ~76% of its PTQ imm price. On the two factor-width
cuts measured since, QAT removed 47% and 90% - so this is a coin flip that a 750-user run can settle for
about 35 min per point, and the two points are already registered in BOTH `quant_sweep.CANDIDATES` and
`check_corruption.CONFIGS`, so nothing else has to change.

WHY BOTH RUNGS AND NOT JUST ONE. They differ by the note's own shift cut (int4 -> int3), which row 35
measured on 750 users as nearly free (+0.000011/+0.000037) and the screen agrees is free, so the second
point is expected to be the better of the two and the 256-bit prize is the larger one. The card-only rung
isolates the card's shift cut, which is the one the screen says costs.

NOT A NEW DIRECTION. Same lineage, same gate, same 124-step decay and the same `qnj_it36ctl` fp32 twin as
chains 17/18/21/22; the candidates were pre-registered on 2026-11-23 for exactly this. If both points fail,
the shift ladder stays closed and the answer is measured rather than inferred.

CORRUPTION PHASE. This chain runs last, so it runs its own 50-user finiteness check for any point whose own
sweep PASSES (ub95 <= +0.0005 on BOTH heads) and writes its end marker only afterwards.

Waits behind `it36e1combo` (chain 24), which waits behind chains 23, 22, 21, 19 and 18. Outside
`autorun/QUEUE.md` on purpose: it is the interactive session's chain, like chains 15-24.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36shiftrung_chain.log"
LOCK = RUN / "it36shiftrung_chain.lock"
PREV_LOG = RUN / "it36e1combo_chain.log"
# the prefix, so it also matches chain 24's "=== it36e1combo chain end (without running anything) ==="
PREV_END = "=== it36e1combo chain end"
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin, gate stamp)
QAT_POINTS = [
    ("l2i6s3n1i6s4", "it36l2i6s3n1i6s4", 124, "qnj_it36ctl", None),
    ("l2i6s3n1i6s3", "it36l2i6s3n1i6s3", 124, "qnj_it36ctl", None),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i6s3n1i6s4", "it36l2i6s3n1i6s4",
     "chain 25 point 1 (card shifts int4 -> int3 under the int6 card, 960/1,920 = 2,880 bits)"),
    ("l2i6s3n1i6s3", "it36l2i6s3n1i6s3",
     "chain 25 point 2 (both streams' shifts at int3, 960/1,728 = 2,688 bits)"),
]

sys.path.insert(0, str(RUN))
import it36ladder_lib as lib  # noqa: E402  the shared wedge watchdog + retry

lib.WAIT_MAX_MIN = WAIT_MAX_MIN
lib.IDLE_PROCEED_MIN = IDLE_PROCEED_MIN


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def running(pattern):
    return lib.running(pattern)


def wait_for_prev_chain():
    t0 = time.time()
    idle = 0
    while True:
        ended = PREV_END in PREV_LOG.read_text(encoding="utf-8") if PREV_LOG.exists() else False
        busy = running(BUSY_PAT)
        if ended and busy == 0:
            log("GPU free: previous chain logged its end marker, 0 matching processes")
            return True
        idle = idle + 1 if busy == 0 else 0
        if idle >= IDLE_PROCEED_MIN:
            log(f"no matching process for {idle} checks and no end marker -- assuming the previous "
                f"chain is gone; proceeding")
            return True
        if (time.time() - t0) / 60 > WAIT_MAX_MIN:
            log(f"GAVE UP waiting after {WAIT_MAX_MIN} min (ended={ended}, busy={busy})")
            return False
        time.sleep(60)
        log(f"waiting: prev chain ended={ended}, matching processes={busy}")


def gate_passed(stamp):
    """True / False / None (no verdict yet). A row is a deploy candidate only on PASS of both heads."""
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text(encoding="utf-8")).get("rows") or []
    except ValueError:
        return None
    if not rows:
        return None
    g = rows[0].get("gate") or {}
    a = (g.get("ahead") or {}).get("ub95")
    i = (g.get("imm") or {}).get("ub95")
    if a is None or i is None:
        return None
    return bool(a <= GATE_UB95 and i <= GATE_UB95)


def main():
    log("=== it36shiftrung chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36shiftrung chain end (without running anything) ===")
        return 1

    for cand, stamp, steps, ref, gate in QAT_POINTS:
        if gate is not None:
            ok = gate_passed(gate)
            if ok is not True:
                log(f"point {cand}: gate {gate} is {ok} -- skipping this point")
                continue
        epochs = max(2, round(steps / 62))
        log(f"point {cand}@{steps} -> tag {stamp}, ref {ref}, epochs {epochs}")
        rc = lib.qat_point(log, cand, stamp, steps, ref, epochs)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    # CORRUPTION PHASE (Andrew's standing order: never leave a corruption check pending).
    for name, stamp, note in COR_POINTS:
        out = COR / f"corruption_corrupt_{name}.json"
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS":
                    log(f"{name}: PASS already recorded -- skipping")
                    continue
            except (KeyError, ValueError):
                pass
        ok = gate_passed(stamp)
        if ok is False:
            log(f"{name}: {stamp} fails the log-loss gate -- not a deploy candidate, no check needed")
            continue
        if ok is None:
            log(f"{name}: no usable sweep verdict for {stamp} yet -- skipping "
                f"(a missing verdict is never a pass)")
            continue
        if not (RUN / f"qat_{stamp}" / "qat_124.pth").exists():
            log(f"{name}: checkpoint missing for {stamp} -- skipping")
            continue
        log(f"--- point {name} ({note}) passes the gate: running the corruption check")
        rc = lib.run(log, f"corrupt {name}",
                     [str(PY), "-u", str(DRIVER), "--tag", name, "--device", "cuda"],
                     COR / f"launcher_{name}.log")
        if rc != 0:
            log(f"{name}: driver rc={rc} (a FAIL or INCOMPLETE verdict is in its JSON)")

    log("=== it36shiftrung chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
