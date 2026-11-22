"""Chain 21: the int6 CARD rung stacked on the note's int3 SHIFT rung -> 3,136 bits.

WHY THIS CHAIN EXISTS. Two measured prices meet at the same 3,136 bits that row 41
(`l3i4s4n8s3`) just missed the imm ceiling by 0.000016:

  * `l2i6c` - the CARD at int6 with the note untouched - is nearly FREE as a per-user PAIRED delta on
    11 long histories: +0.000052 ahead [-0.000007, +0.000109] and +0.000115 imm [+0.000018, +0.000218]
    (`rust_wide_screen_it36int6split.json`, GPU_RUN_BOARD Amendment 22). It is DEPLOYABLE because the
    engine's `parse_level` already accepts int6.
  * row 35's note int3 SHIFT rung (`n1i8s3`) costs +0.000011 ahead / +0.000037 imm for 192 fewer note
    bits, and row 35 is the smallest PASSING row at 3,392 bits.

Stacking them: card rank-2 int6 + int4 shifts = 1,024 bits, note rank-1 int8 + int3 shifts = 2,112
bits, total 3,136 - 256 BELOW row 35 and 448 below row 17 (3,584). Row 41 reached the same size the
other way, with rank-3 int4 card factors, whose own imm price (+0.000472, row 33) is 4x the int6 card's.
`l2i6s4n1i8s3` is pre-registered in `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and
`check_corruption.CONFIGS`; `rust_grid_it36int6shift.json` + `wide_screen.py` price it on the same long
histories as the CPU-side screen that priced int6.

CORRUPTION PHASE. Andrew's standing order is that a corruption check follows a passing row immediately,
and this chain is LAST in run order, so `it36corrupt2` cannot pick the row up. It therefore runs its own
50-user check for the point if its own sweep PASSES (ub95 <= +0.0005 on BOTH heads), and writes its end
marker only afterwards.

Waits behind `it36fixedbasis` (chain 19), which itself waits behind `it36int6card` (chain 18), so the
`l2i6c` verdict this point depends on lands first. Outside `autorun/QUEUE.md` on purpose: it is the
interactive session's chain, like `it36e1`, `it36e1shrink`, `it36int6`, `it36int6card` and
`it36fixedbasis`.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36int6combo_chain.log"
LOCK = RUN / "it36int6combo_chain.lock"
PREV_LOG = RUN / "it36fixedbasis_chain.log"
PREV_END = "=== it36fixedbasis chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin) - the QAT point
QAT_POINTS = [
    ("l2i6s4n1i8s3", "it36l2i6s4n1i8s3", 124, "qnj_it36ctl"),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i6s4n1i8s3", "it36l2i6s4n1i8s3", "chain 21 point 1 (int6 card + note int3 shifts, 3,136 bits)"),
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
    log("=== it36int6combo chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36int6combo chain end (without running anything) ===")
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

    log("=== it36int6combo chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())