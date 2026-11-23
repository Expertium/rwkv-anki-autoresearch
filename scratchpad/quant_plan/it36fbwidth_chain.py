"""Chain 27: TWO NEW codec knobs - the m x m coefficient block's own BIT-WIDTH, and the CARD on the codec.

WHY THIS CHAIN EXISTS. The codec's payload per layer is m*m*qbits + 64*shift_bits, so it has a size
knob the uniform (rank, factor-width, shift-width) design does not: the precision of the COEFFICIENT
block itself. `fake_fixedbasis_state` quantizes that block with the same qmax the rank-r factors use,
and the level rides in the LOWRANK scope (`note:6:int6` = m=6 with int6 coefficients), so no new code
is needed - the RUST engine reads the same level from its FIXEDBASIS scope (`model.rs:2423`), and every
level from int2 to int8 already exists in `parse_level` / `quant_sweep._QMAX`.

At m=6 int8 (row 51, 2,912 bits) the codec BEAT row 45's 2,944-bit uniform frontier on both heads
(+0.000018/+0.000209 vs +0.000125/+0.000381) with a LOWER held-out capture (0.82-0.87 vs 0.904-0.951),
so the rank-1-note truncation is not what the loss is paying for at these sizes. That leaves two cheap
questions this chain answers with four points:

  * is the coefficient block's int8 precision even needed? int6 keeps 64 levels per (card, note, layer)
    matrix and saves 216 bits on the note (2,912 -> 2,696); int4 (16 levels) saves 432 (-> 2,480).
  * can the CARD leave the rank-2 family entirely? The basis file already carries `card 0 16`
    (P__card__L0 / Q__card__L0, held-out capture 0.9195 at m=8 / 0.9596 at m=16 across users), so a
    card codec at m=8 costs 1*(64*8 + 64*4) = 768 bits against row 17's rank-2 card's 1,280 - a 512-bit
    saving on the stream that owns the immediate-head tax, at a rank-8 (not rank-2) fit of its state.

THE FOUR POINTS (124 decay steps each against `qnj_it36ctl`, ungated point-on-point, so a single broken
rung does not hide the rest - the same shape as chain 23's E1 curve and chain 26's m ladder):
  1. `l2i8fb6n6` card rank-2 int8 + note codec m=6 int6   1,280/1,416 = 2,696 bits (248 below row 45)
  2. `l2i8fb6n4` card rank-2 int8 + note codec m=6 int4   1,280/1,200 = 2,480 bits (464 below row 45)
  3. `fbc8n6i8`  card codec m=8 int8 + note codec m=6 int8  768/1,632 = 2,400 bits (544 below row 45)
  4. `fbc8n6i6`  card codec m=8 int8 + note codec m=6 int6  768/1,416 = 2,184 bits (760 below row 45)
Points 1/2 isolate the coefficient width (the card stays row 17's); 3 isolates the card (the note stays
at row 51's known-good m=6 int8); 4 is the combination, so it is last.

SPEED IS ALREADY MEASURED (CPU-only engine probe, `rust_speed_c27.json`, 2026-11-23 10:02, run beside a
chain job so the ratios read LOW): `plain` 2,812.8 rev/s, l2i8fb6n (the note-only codec reference) 0.792x
rev/s / 0.781x median per-user, l2i8fb6n6 0.770x / 0.760x, l2i8fb6n4 0.792x / 0.755x, fbc8n6i8 0.877x /
0.868x, fbc8n6i6 0.864x / 0.856x - ALL FOUR above the 0.75x floor, and the two BOTH-CODEC configs are
the FASTEST of the family, because replacing the card's rank-2 per-step deflation with two small m=8
matmuls is cheaper than the Gram/eigen path it replaces. So axes 1 and 3 both improve here.

CORRUPTION PHASE. This chain runs its own 50-user finiteness check for any point whose own sweep PASSES
the log-loss gate (ub95 <= +0.0005 on BOTH heads) and writes its end marker only afterwards, so the next
chain cannot start while a check is still running. Andrew's standing order: never leave a corruption
check pending.

Waits behind `it36fbrung` (chain 26), the tail of the live queue when this was written. Outside
`autorun/QUEUE.md` on purpose: it is the interactive session's chain, like `it36e1`, `it36int6`,
`it36int6card`, `it36fixedbasis`, `it36fbrung` and the rest.
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
LOG = RUN / "it36fbwidth_chain.log"
LOCK = RUN / "it36fbwidth_chain.lock"
PREV_LOG = RUN / "it36fbrung_chain.log"
PREV_END = "=== it36fbrung chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin) - the QAT points
QAT_POINTS = [
    ("l2i8fb6n6", "it36l2i8fb6n6", 124, "qnj_it36ctl"),
    ("l2i8fb6n4", "it36l2i8fb6n4", 124, "qnj_it36ctl"),
    ("fbc8n6i8", "it36fbc8n6i8", 124, "qnj_it36ctl"),
    ("fbc8n6i6", "it36fbc8n6i6", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8fb6n6", "it36l2i8fb6n6", "chain 27 point 1 (codec note m=6 int6 coefficients, 2,696 bits)"),
    ("l2i8fb6n4", "it36l2i8fb6n4", "chain 27 point 2 (codec note m=6 int4 coefficients, 2,480 bits)"),
    ("fbc8n6i8", "it36fbc8n6i8", "chain 27 point 3 (codec on the CARD too, card m=8 int8 + note m=6 int8, 2,400 bits)"),
    ("fbc8n6i6", "it36fbc8n6i6", "chain 27 point 4 (codec on the CARD too, card m=8 int8 + note m=6 int6, 2,184 bits)"),
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
    log("=== it36fbwidth chain start ===")
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
    log(f"the codec coefficient-width + card-codec ladder: {[p[0] for p in QAT_POINTS]}, "
        f"waiting behind {PREV_LOG.name}")
    if not wait_for_prev_chain():
        log("=== it36fbwidth chain end (without running anything) ===")
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

    log("=== it36fbwidth chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
