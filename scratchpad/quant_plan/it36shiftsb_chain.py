"""Chain 28: the TOKEN-SHIFT CODEC - the note's two persisted shift vectors at m coefficients.

WHY THIS CHAIN EXISTS. The persisted token-shift vectors are 2*32 = 64 numbers per layer and, at int4,
2*64*4 = 256 of every compressed note layer's 768 bits - 47% of the note's payload. They are LayerNorm
outputs with a near-SHARED mean, so a MODEL-SIDE (mu, P) - 32 + 32*m floats, which cost ZERO state bits -
can code them as m coefficients at the stream's shift level: 2*m*level bits per layer.

MEASURED on the 6-user shift dump (3,000 rows per key, `shift_probe_out/shifts_note_multi6.npz`,
token-shift role 0, 9,000 rows over 3 note layers), relative reconstruction error:
  m=16 int4   64 bits/vec  0.01119   vs raw int4 128 bits/vec  0.01214   <- HALF the bits, LESS error
  m=16 int6   96 bits/vec  0.00664   vs raw int4 128 bits/vec  0.01214
  m=16 int8  128 bits/vec  0.00641   vs raw int4 128 bits/vec  0.01214   <- same bits, 1.9x less error
  m=12 int4   48 bits/vec  0.01568
  m=8  int4   32 bits/vec  0.02486
Mean subtraction alone (`mean int4`, 128 bits/vec) is 0.00525 - better error than the codec at m=16 int4
but TWICE the bits; bits are axis 1, so the codec is the right pick. The SINGLE-USER numbers that first
motivated this (m=8 int8 = 0.0132) were optimistic: pooled over 6 users m=8 int8 = 0.0244, WORSE than raw
int4 at half the bits, so m=8 is only on the ladder as the low end, not as the win.

THE CARD IS DELIBERATELY LEFT ALONE. Its shift subspace is much harder: m=16 int8 (128 bits/vec) only
MATCHES raw int4's 0.01738 (0.01701), i.e. the codec buys the card nothing at any width. The artifact is
therefore NOTE-ONLY (its bare `r0`/`r1` tiers were dropped so a card lookup cannot silently hit a
NOTE-fitted basis, whose cross-stream error is 0.10-0.25) and every candidate sets
RWKV_SHIFT_FIXEDBASIS_SCOPE=note, which the Python mixer, the Rust engine and BOTH bit accounts honour.

THE FIVE POINTS (124 decay steps each against `qnj_it36ctl`, ungated point-on-point like chains 23/26/27).
Every point holds chain 26's PASSING point 1 (`l2i8fb4n`: card rank-2 int8 + note state codec m=4 int8,
1,280/1,152 = 2,432 bits) and changes ONLY the note's shift coder:
  1. `l2i8fb4nsb16i4` note shift codec m=16 int4   1,280/768  = 2,048 bits (896 below row 45)
  2. `l2i8fb4nsb16i6` note shift codec m=16 int6   1,280/960  = 2,240 bits (704 below row 45)
  3. `l2i8fb4nsb16i8` note shift codec m=16 int8   1,280/1,152 = 2,432 bits (ERROR-ONLY A/B, same bits
     as point 0: if the loss is flat here, the codec's error is not what the tax is paying for)
  4. `l2i8fb4nsb12i4` note shift codec m=12 int4   1,280/672  = 1,952 bits (992 below row 45)
  5. `l2i8fb4nsb8i4`  note shift codec m=8 int4    1,280/576  = 1,856 bits (1,088 below row 45)
Points 1/4/5 are the size ladder (m at int4), 2 is the coefficient-width step, 3 is the same-bits control.

SPEED (axis 3, low priority): the codec replaces a 32-coordinate int-N round-trip with two small matmuls
(32xm and mx32) per vector, and its coefficient block is TINY compared with the rank-2 card path it sits
next to. `rust_speed_c28.json` records the engine probe for these five configs (plain reference plus
`l2i8fb4n`) against the 0.75x rev/s floor.

CORRUPTION PHASE. This chain runs its own 50-user finiteness check for any point whose own sweep PASSES
the log-loss gate (ub95 <= +0.0005 on BOTH heads) and writes its end marker only afterwards, so the next
chain cannot start while a check is still running. Andrew's standing order: never leave a corruption
check pending.

Waits behind `it36fbwidth` (chain 27), the tail of the live queue when this was written. Outside
`autorun/QUEUE.md` on purpose: it is the interactive session's chain, like `it36e1`, `it36int6`,
`it36fixedbasis`, `it36fbrung`, `it36fbwidth` and the rest.
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
LOG = RUN / "it36shiftsb_chain.log"
LOCK = RUN / "it36shiftsb_chain.lock"
PREV_LOG = RUN / "it36fbwidth_chain.log"
PREV_END = "=== it36fbwidth chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin) - the QAT points
QAT_POINTS = [
    ("l2i8fb4nsb16i4", "it36l2i8fb4nsb16i4", 124, "qnj_it36ctl"),
    ("l2i8fb4nsb16i6", "it36l2i8fb4nsb16i6", 124, "qnj_it36ctl"),
    ("l2i8fb4nsb16i8", "it36l2i8fb4nsb16i8", 124, "qnj_it36ctl"),
    ("l2i8fb4nsb12i4", "it36l2i8fb4nsb12i4", 124, "qnj_it36ctl"),
    ("l2i8fb4nsb8i4", "it36l2i8fb4nsb8i4", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8fb4nsb16i4", "it36l2i8fb4nsb16i4", "chain 28 point 1 (note shift codec m=16 int4, 2,048 bits)"),
    ("l2i8fb4nsb16i6", "it36l2i8fb4nsb16i6", "chain 28 point 2 (note shift codec m=16 int6, 2,240 bits)"),
    ("l2i8fb4nsb16i8", "it36l2i8fb4nsb16i8", "chain 28 point 3 (note shift codec m=16 int8, same-bits control, 2,432 bits)"),
    ("l2i8fb4nsb12i4", "it36l2i8fb4nsb12i4", "chain 28 point 4 (note shift codec m=12 int4, 1,952 bits)"),
    ("l2i8fb4nsb8i4", "it36l2i8fb4nsb8i4", "chain 28 point 5 (note shift codec m=8 int4, 1,856 bits)"),
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
    log("=== it36shiftsb chain start ===")
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
    log(f"the token-shift codec ladder: {[p[0] for p in QAT_POINTS]}, "
        f"waiting behind {PREV_LOG.name}")
    if not wait_for_prev_chain():
        log("=== it36shiftsb chain end (without running anything) ===")
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

    log("=== it36shiftsb chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
