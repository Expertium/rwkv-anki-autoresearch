"""Chain 23: the E1 tail-energy penalty, second attempt - now that it fits in the VRAM cap.

WHAT CHANGED, AND WHY THE FIRST ATTEMPT IS NOT A REFUTATION. Chain 15 (`it36e1`) queued three E1
points: the fused-off control `l2i8s4pf` and two penalty points (`l2i8s4e1a` lambda=1.0,
`l2i8s4e1b` lambda=10.0). The control PASSED and is RESULTS row 44 (+0.000050/+0.000325 against
`qnj_it36ctl`, and within 2e-6 of row 17's fused run, which is the parity the arm needed). Both
penalty points died with CUDA OOM at 10.34 GiB of the 11.00 GiB cap BEFORE their first training step.

The cause was measured, not guessed: `accum_tail_penalty` runs a DIFFERENTIABLE `svdvals` once per
TIMESTEP per rank-truncated layer, and every one of those SVD graphs stays alive until backward, so
the penalty's memory grows with the training WINDOW and it landed on top of an already-full 10.34 GiB
card. It was NOT a broken penalty: `corruption/e1_cpu_smoke.py` (CPU, 2,048-token batch, 3 steps)
PASSED with `[tail-penalty] lambda=1.0 card=0.002918(n=23) note=0.038505(n=102)`.

THE FIX (2026-11-23, `rwkv_ops.py`): the penalty accumulates on ONE step in `RWKV_QAT_TAIL_STRIDE`
per stream (default 8). The mean over the sampled steps is an unbiased estimate of the mean tail
over the window, at 1/stride of the graph memory. Verified in two steps before this chain was
written: `scratchpad/_test_e1_stride.py` (stride 4 -> 2 of 8 calls, stride 8 -> 1 of 8, stride 1 ->
8 of 8, counter is per stream tag) and the real trainer on CPU
(`corruption/e1_cpu_smoke.py` with `E1SMOKE_STRIDE=8`).

WHY THIS ARM IS THE ONLY ZERO-BIT LEVER LEFT. The deploy path stores a rank-r matrix per stream, and
the gate's measured tax is RANK TRUNCATION, not precision: `cn8` (full-matrix int8, 4x over the bars,
diagnostic only) scores +0.000033/+0.000022, so an 8-bit state with no truncation is lossless. The
measured tails of row 17's own states are 0.62% (card, rank 2) and 3.06% (note, rank 1). A state
whose energy is EXACTLY rank-r loses nothing to the truncation, and the QAT loss cannot see the tail
at all, because the straight-through estimator passes gradient through the truncation as if it were
identity. So E1 adds `lambda * mean over streams of the tail fraction` to the training loss: zero
state bits, zero deploy cost. Andrew's axis-2 target.

READING THE RESULT. Every point is 124 steps, the same length as row 17 and row 44, against the same
`qnj_it36ctl` twin, so a delta is comparable with row 17 (+0.000052/+0.000327) and row 44
(+0.000050/+0.000325). Three points bracket the lambda curve (1.0 ~ 2.4% of the ~1.02 loss,
3.0 ~ 7%, 10.0 ~ 25%). NO gate between them: the three lambdas are three points on one curve, and a
FAIL at a small lambda says the penalty is too weak, not that the arm is dead, so gating them on each
other would throw away the measurement that matters. A FAIL at ALL THREE says the deploy tax is not
tail energy - a real result that closes the arm.

A point whose 0.04-epoch smoke OOMs is SKIPPED by `qat_point` (that is what happened in chain 15), so
a memory regression here is self-diagnosing and costs one smoke, not one eval.

CORRUPTION PHASE. Andrew's standing order: a corruption check follows a passing row immediately, and
this chain runs last, so it runs its own 50-user check (with `RWKV_QAT_FUSED=0`, the bit-parity
Python twin, the only path where the state is visible) for every point whose sweep PASSES. The check
eval env is `l2i8s4pf`'s (identical scope, no penalty); the penalty is a training-only lever and is
inert at eval, since an eval pass has an empty stream tag.

Waits behind `it36int6note` (chain 22), which waits behind `it36int6combo`, `it36fixedbasis`,
`it36int6card` and `it36int6`. Outside `autorun/QUEUE.md` on purpose: it is the interactive
session's chain, like chains 15-22, so the hourly supervisor must not start, kill, restart or score
it.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36e1b_chain.log"
LOCK = RUN / "it36e1b_chain.lock"
PREV_LOG = RUN / "it36int6note_chain.log"
PREV_END = "=== it36int6note chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005

# (candidate, sweep stamp, decay steps, fp32 twin, gate stamp)
QAT_POINTS = [
    ("l2i8s4e1c", "it36l2i8s4e1c", 124, "qnj_it36ctl", None),
    ("l2i8s4e1d", "it36l2i8s4e1d", 124, "qnj_it36ctl", None),
    ("l2i8s4e1e", "it36l2i8s4e1e", 124, "qnj_it36ctl", None),
]
# (corruption config name, the sweep stamp that owns the gate, what the row is)
COR_POINTS = [
    ("l2i8s4e1c", "it36l2i8s4e1c",
     "chain 23 point 1 (E1 tail penalty lambda=1.0, stride 8, on row 17's 1,280/2,304 layout)"),
    ("l2i8s4e1d", "it36l2i8s4e1d",
     "chain 23 point 2 (E1 tail penalty lambda=3.0, stride 8, on row 17's 1,280/2,304 layout)"),
    ("l2i8s4e1e", "it36l2i8s4e1e",
     "chain 23 point 3 (E1 tail penalty lambda=10.0, stride 8, on row 17's 1,280/2,304 layout)"),
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
    log("=== it36e1b chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36e1b chain end (without running anything) ===")
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

    log("=== it36e1b chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
