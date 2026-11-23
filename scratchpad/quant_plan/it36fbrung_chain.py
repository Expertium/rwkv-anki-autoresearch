"""Chain 26: the FIXED-BASIS (codec) size ladder - the note's coefficient block at m=4 and m=5.

WHY THIS CHAIN EXISTS. Rows 50 and 51 (chain 19) measured the codec at m=8 and m=6 and BOTH passed
the loss gate, at row 17's exact size (3,584 bits) and 32 bits below the uniform-design frontier
(2,912 vs 2,944). At a FIXED size the codec BEAT the per-step rank-r factors on both heads
(+0.000040/+0.000129 vs +0.000052/+0.000327), and at 2,912 bits it beat row 45's own tax
(+0.000018/+0.000209 vs +0.000125/+0.000381) even though its held-out capture is LOWER (0.82-0.87 vs
0.904-0.951). So capture does not predict the loss, and the codec's m is a size axis the uniform
design does not have: the payload is m*m*qbits for the whole WKV rather than 2*K*r*qbits.

THE LADDER. One fitting artifact (`corruption/fixedbasis_l2i8s4all.npz` / `.txt`, fitted at 16 modes,
sliced to its first m columns - the separable 2D-PCA basis is nested, so no re-fit is needed):
  * `l2i8fb4n`: card rank-2 int8 + note codec m=4 int8 = 1,280/1,152 = 2,432 bits - 512 BELOW row 45's
    2,944-bit frontier, held-out capture 0.751-0.856 per layer;
  * `l2i8fb5n`: card rank-2 int8 + note codec m=5 int8 = 1,280/1,368 = 2,648 bits, capture between
    m=4's and m=6's (0.818-0.871).
Both hold row 17's card and its int4 shifts, so each is a clean note-only A/B against rows 17/50/51.
The ladder is NOT gated point-on-point: if m=4 breaks the gate, m=5 (the next size up) is still the
point that would fill the gap, exactly like chain 23's ungated E1 curve.

MILEAGE. Both points run 124 decay steps against `qnj_it36ctl` (every codec row so far used 124, so the
comparison stays apples-to-apples) - smoke, qat, then the 750-user roster sweep.

CORRUPTION PHASE. This chain runs its own 50-user finiteness check for any point whose own sweep PASSES
the log-loss gate (ub95 <= +0.0005 on BOTH heads) and writes its end marker only afterwards, so the
next chain cannot start while a check is still running. Andrew's standing order: never leave a
corruption check pending.

Waits behind `it36shiftrung` (chain 25), which is the tail of the live queue. Outside
`autorun/QUEUE.md` on purpose: it is the interactive session's chain, like `it36e1`, `it36int6`,
`it36int6card`, `it36fixedbasis` and the rest.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36fbrung_chain.log"
LOCK = RUN / "it36fbrung_chain.lock"
PREV_LOG = RUN / "it36shiftrung_chain.log"
PREV_END = "=== it36shiftrung chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin) - the QAT points
QAT_POINTS = [
    ("l2i8fb4n", "it36l2i8fb4n", 124, "qnj_it36ctl"),
    ("l2i8fb5n", "it36l2i8fb5n", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8fb4n", "it36l2i8fb4n", "chain 26 point 1 (codec note m=4, 2,432 bits)"),
    ("l2i8fb5n", "it36l2i8fb5n", "chain 26 point 2 (codec note m=5, 2,648 bits)"),
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
    log("=== it36fbrung chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    log(f"the codec m ladder: {[p[0] for p in QAT_POINTS]}, waiting behind {PREV_LOG.name}")
    if not wait_for_prev_chain():
        log("=== it36fbrung chain end (without running anything) ===")
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

    log("=== it36fbrung chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
