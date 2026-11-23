"""Chain 29: the FIXED-BASIS CODEC's mode ladder DOWNWARD (m=3 and m=2).

WHY THIS CHAIN EXISTS. The codec stores a compressed layer's state as an m x m coefficient block against
a basis that ships with the model, so its payload is m*m*qbits bits per layer instead of the rank-r path's
2*K*r*qbits. The ladder so far walked UP from m=4 and stayed flat: m=4 int8 measures 1,152 note bits and
+0.000032/+0.000311 (row 61, the smallest passing codec row), m=5 (row 62) and m=6 int6 (row 63) pass too.
The low end was never tried, and it is the end that BUYS state bits: the note payload is
3*(m*m*qbits + 64*shift_bits), so at the shipped int8 coefficient width

  m=4 -> 3*(16*8 + 256) = 1,152 bits      (row 61, passing)
  m=3 -> 3*( 9*8 + 256) =   984 bits
  m=2 -> 3*( 4*8 + 256) =   864 bits
  m=2 int4 -> 3*(4*4 + 256) = 816 bits    (the floor of this family on the raw int4 shifts)

Against that, the held-out basis capture falls as m drops (m=4 captures 0.751-0.856 of a note layer's
energy, m=6 0.818-0.871), which is exactly the trade this chain prices. Note what the two int4 rungs imply:
once the state payload is 96 bits per layer the note's SHIFT vectors (2*32 coordinates at int4 = 256 bits)
are 73% of the note again - which is what chain 28's token-shift codec attacks, so the stacking of this
ladder with chain 28's shift codec is the natural follow-up IF these rungs pass.

THE POINTS (124 decay steps each against `qnj_it36ctl`, ungated point-on-point like chains 23/26/27/28).
Every point holds row 61's card (rank-2 int8 + int4 shifts = 1,280 bits) and changes ONLY the note codec:
  1. `l2i8fb3n`   note codec m=3 int8   1,280/984 = 2,264 bits (168 below row 61)
  2. `l2i8fb2n`   note codec m=2 int8   1,280/864 = 2,144 bits (288 below row 61)
Two points, both at the SHIPPED int8 coefficient width, so the only variable is the mode count. The int4
coefficient rungs (`l2i8fb3n4`, `l2i8fb2n4`) were registered in both bit accounts and in the corruption
driver but are NOT queued: chain 27's point 2 (`l2i8fb6n4`, m=6 int4) failed the immediate head at ub95
+0.000596 against the +0.0005 ceiling while m=6 int6 passed at +0.000294, so int6 is the knee of the
codec's coefficient width and an int4 rung would spend a GPU hour to re-measure that knee.

CORRUPTION PHASE. Any point whose own sweep PASSES the log-loss gate (ub95 <= +0.0005 on BOTH heads) gets
its own 50-user finiteness check before this chain writes its end marker, so the next chain cannot start
while a check is still running (Andrew's standing order).

Waits behind `it36shiftsb` (chain 28). Outside `autorun/QUEUE.md` on purpose: it is the interactive
session's chain, like `it36e1`, `it36int6`, `it36fbrung`, `it36fbwidth`, `it36shiftsb` and the rest.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36fblowm_chain.log"
LOCK = RUN / "it36fblowm_chain.lock"
PREV_LOG = RUN / "it36shiftsb_chain.log"
PREV_END = "=== it36shiftsb chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption|bucket_tax")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin)
QAT_POINTS = [
    ("l2i8fb3n", "it36l2i8fb3n", 124, "qnj_it36ctl"),
    ("l2i8fb2n", "it36l2i8fb2n", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8fb3n", "it36l2i8fb3n", "chain 29 point 1 (note codec m=3 int8, 2,264 bits)"),
    ("l2i8fb2n", "it36l2i8fb2n", "chain 29 point 2 (note codec m=2 int8, 2,144 bits)"),
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
    log("=== it36fblowm chain start ===")
    # ATOMIC lock (os.O_EXCL): the shell tool can run one command twice concurrently, so a plain
    # `if LOCK.exists()` check can let two copies past it. O_EXCL makes the create the arbiter.
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = LOCK.read_text(encoding="utf-8").strip() if LOCK.exists() else "?"
        log(f"lock found ({existing}) -- another copy is going; exiting")
        return 1
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"lock taken: {LOCK}")
    log(f"the codec's low-m ladder: {[p[0] for p in QAT_POINTS]}, waiting behind {PREV_LOG.name}")
    if not wait_for_prev_chain():
        log("=== it36fblowm chain end (without running anything) ===")
        return 1

    for cand, stamp, steps, ref in QAT_POINTS:
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

    log("=== it36fblowm chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
