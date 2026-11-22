"""Chain 22: the NOTE's int6 factor rung stacked on the note's int3 SHIFT rung -> 2,752 bits.

WHY THIS CHAIN EXISTS. The 7-user deploy screen (`rust_grid.py --tag it36wide5`, row-17 deploy
weights, the same 7 held-out long histories every earlier screen used) priced four sub-3,136 rungs in
one run, and this is the only one whose price sits INSIDE the +0.0005 ceiling on BOTH heads:

    config          card/note bits   d ahead    d imm    (paired vs `l2i8s4` = 0.407618/0.395728)
    l2i6s4n1i8s3    1,024 / 2,112    +0.000173  +0.000219   <- chain 21's 3,136-bit point
    l2i5s4n1i8s3      896 / 2,112    +0.000403  +0.000803
    l2i6s4n1i6s3    1,024 / 1,728    +0.000343  +0.000435   <- THIS CHAIN, 2,752 bits
    l2i5s4n1i8s2      896 / 1,920    +0.000162  +0.002081
    l2i6s4n1i6s2    1,024 / 1,536    +0.000104  +0.001642

Two side findings close neighbouring rungs: the int5 CARD is WORSE than int6 on both heads (not a free
step below int6, whatever a 2-user probe said), and the note's int2 SHIFT rung costs +0.0012-0.0014 imm
at either factor width, so it is a dead end for the immediate head.

So the ladder below row 35 (`n1i8s3`, 3,392 bits, the smallest FULLY deployable row) is now:
row 43's 2,880 (FAIL, imm 1.22x over), chain 21's 3,136 and gated 2,944, and this 2,752. The note's
int8 -> int6 factor cut costs only +0.000170 ahead / +0.000216 imm for 384 note bits on this screen,
i.e. about the same as the card's int8 -> int6 cut for 256 card bits, so the two coarsenings together
are the cheapest 832 bits on offer.

CAVEAT, recorded because it decides how to read a failure: this is PTQ against weights trained for the
rank-1-int8 note, so QAT is the one regime the screen cannot see. That is exactly what a 750-user QAT
run answers.

AND NOTE THE WIDTH, not just the mean: with 2,000 bootstrap resamples over the 7 users the intervals are
+0.000343 [+0.000229, +0.000492] ahead and +0.000435 [+0.000188, +0.000722] imm
(`rust_wide_screen_it36wide5.json`). So the imm POINT estimate is inside the +0.0005 ceiling but its 95%
upper bound is not, and this is a marginal rung by the same rule that rejected the 3,136 point's nearest
neighbour - whereas chain 21's 3,136 point reads +0.000219 [-0.000069, +0.000509] and is at the ceiling.
Read a FAIL here as "the note's int6 factors cost more under QAT than PTQ", not as "the rung is wrong".

CORRUPTION PHASE. Andrew's standing order is that a corruption check follows a passing row immediately,
and this chain runs after everything else, so `it36corrupt2` cannot pick the row up. It therefore runs
its own 50-user check if its own sweep PASSES (ub95 <= +0.0005 on BOTH heads), and writes its end
marker only afterwards.

Waits behind `it36int6combo` (chain 21), which itself waits behind `it36fixedbasis`, `it36int6card`,
`it36int6` and `it36e1shrink`. Outside `autorun/QUEUE.md` on purpose: it is the interactive session's
chain, like chains 15-21.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36int6note_chain.log"
LOCK = RUN / "it36int6note_chain.lock"
PREV_LOG = RUN / "it36int6combo_chain.log"
PREV_END = "=== it36int6combo chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin, gate stamp)
QAT_POINTS = [
    ("l2i6s4n1i6s3", "it36l2i6s4n1i6s3", 124, "qnj_it36ctl", None),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i6s4n1i6s3", "it36l2i6s4n1i6s3",
     "chain 22 point 1 (int6 card + note int6 factors + note int3 shifts, 2,752 bits)"),
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
    log("=== it36int6note chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36int6note chain end (without running anything) ===")
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

    log("=== it36int6note chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())