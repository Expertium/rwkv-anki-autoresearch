"""Chain 19: the FIXED-BASIS (codec) state-coding points, then their corruption checks.

WHY THIS CHAIN EXISTS. Andrew approved approach 1 on 2026-11-22 20:51 ("Alright, try 1"): store the
m x m coefficient block C = P^T W Q against a FIXED basis that ships with the model, instead of the
rank-r path's per-step 2*K*r factors. Implemented in BOTH languages
(`rwkv/model/rwkv_ops.py::fake_fixedbasis_state`, `rust/rwkv-infer/src/model.rs::fixedbasis_roundtrip`),
unit-tested (10/10 Python checks, 4/4 Rust tests), and its basis is fitted on 3 users of the row-17
checkpoint (`corruption/fixedbasis_l2i8s4all.npz`, exported to `.txt` for the engine).

THE MEASUREMENT THAT SIZES THE CLAIM (`corruption/fixedbasis_l2i8s4all.json`, LEAVE-ONE-USER-OUT).
The in-sample curve is optimistic - a basis fitted on the same states it is scored on captures 0.9995
at m=8 - but held out it captures only 0.876-0.921 (m=8) and 0.82-0.87 (m=6). Against that:
  * the CARD is EXACT at rank 2 for 1,280 bits, while the best fixed basis on the card needs m=16
    (2,304 bits) to reach 0.9596 held out, so the card keeps the rank-r path here;
  * the NOTE's shipped rank-1 captures 0.904-0.951 per layer and PASSES the gate (row 17), so a codec
    reaching 0.88-0.92 for the SAME 768 bits/layer is worth a real log-loss measurement;
  * `l2i8fb8n` = card rank-2 int8 + note fb m=8 = 3,584 bits = row 17's EXACT size -> a pure codec A/B;
  * `l2i8fb6n` = card rank-2 int8 + note fb m=6 = 2,912 bits = 672 BELOW row 17 -> the size claim.

A NOTE ON THE RUST SIDE: the deploy engine can already run both (a 2-user paired screen,
`rust_grid_it36fbsmoke.json`, shows sane small deltas, not a broken codec), but that screen is 2 users
and cannot rank configs - the method lesson of this lineage.

CORRUPTION PHASE. This chain is LAST in run order, so `it36corrupt2` (which guards the earlier rows)
cannot pick these up. It therefore runs its own 50-user corruption check for any point whose own sweep
PASSES the log-loss gate (ub95 <= +0.0005 on BOTH heads), and writes the end marker only afterwards,
so the next chain cannot start while a check is still running.

Waits behind `it36int6card` (chain 18). Outside `autorun/QUEUE.md` on purpose: it is the interactive
session's chain, like `it36e1`, `it36e1shrink`, `it36int6` and `it36int6card`.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36fixedbasis_chain.log"
LOCK = RUN / "it36fixedbasis_chain.lock"
PREV_LOG = RUN / "it36int6card_chain.log"
PREV_END = "=== it36int6card chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin) - the QAT points
QAT_POINTS = [
    ("l2i8fb8n", "it36l2i8fb8n", 124, "qnj_it36ctl"),
    ("l2i8fb6n", "it36l2i8fb6n", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8fb8n", "it36l2i8fb8n", "chain 19 point 1 (fixed-basis note m=8)"),
    ("l2i8fb6n", "it36l2i8fb6n", "chain 19 point 2 (fixed-basis note m=6)"),
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
    log("=== it36fixedbasis chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36fixedbasis chain end (without running anything) ===")
        return 1

    for cand, stamp, steps, ref in QAT_POINTS:
        epochs = max(2, round(steps / 62))
        log(f"point {cand}@{steps} -> tag {stamp}, ref {ref}, epochs {epochs}")
        rc = lib.qat_point(log, cand, stamp, steps, ref, epochs)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    # CORRUPTION PHASE (Andrew's standing order: never leave a corruption check pending). Only a row
    # whose own sweep passes the log-loss gate is a deploy candidate and needs the check.
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

    log("=== it36fixedbasis chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
