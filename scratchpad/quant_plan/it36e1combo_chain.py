"""Chain 24: spend the E1 tail penalty on the axis-1 frontier (row 45, 2,944 bits).

WHY. Row 45 (`l2i6s4`) PASSES the log-loss gate but its immediate head sits at 0.89x of the +0.0005
ceiling (ub95 +0.000446), so there is no room for the next 128-bit cut. E1 - the tail-energy penalty -
is the only zero-bit, zero-deploy-cost lever left, and it is being measured by chain 23 on row 17's
layout (3,584 bits, `l2i8s4e1c/d/e` with a penalty-free control `l2i8s4pf`).

THE SELECTION RULE, and why a plain PASS is not enough. At 3,584 bits every variant passes the
ceiling trivially - the control itself reads +0.000050/+0.000325 - so "pass" says nothing about
whether the penalty helped. What "E1 helps" MEANS here is: the penalty's own deploy deltas beat the
penalty-free control's deltas on BOTH heads. This chain reads chain 23's three sweeps plus the
control, keeps the lambdas that beat the control on both heads, and runs the COMBINATION for the best
of them (lowest mean of the two deltas) on row 45's layout. If none beats the control, E1 does not
buy margin where it matters here and the chain runs nothing - which is itself the verdict, and it
costs no GPU time.

WHY ONLY ONE COMBINATION POINT. The three lambdas are one curve; the best point on it is the one
worth spending 2,944 bits on, and the other two combinations stay registered
(`l2i6s4e1c/d/e` in `quant_sweep.CANDIDATES`) for a follow-up if the winner's own delta is close to
the ceiling.

CORRUPTION PHASE. A passing combination is a deploy candidate, so it gets its own 50-user finiteness
check (`RWKV_QAT_FUSED=0`, the only path where the state is visible) before this chain logs its end
marker, and the eval env is `l2i6s4`'s - the penalty is a training-only lever and is inert at eval.

Waits behind `it36e1b` (chain 23), which waits behind `it36int6note` and the rest. Outside
`autorun/QUEUE.md` on purpose: it is the interactive session's chain, like chains 15-23, so the hourly
supervisor must not start, kill, restart or score it.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36e1combo_chain.log"
LOCK = RUN / "it36e1combo_chain.lock"
PREV_LOG = RUN / "it36e1b_chain.log"
PREV_END = "=== it36e1b chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 2400
IDLE_PROCEED_MIN = 12
GATE_UB95 = 0.0005
CONTROL_STAMP = "it36l2i8s4pf"          # chain 23's penalty-free control, same 3,584 bits, same length
STEPS = 124
REF = "qnj_it36ctl"

# (chain-23 lambda sweep, the matching combination candidate on row 45's layout, lambda)
COMBOS = [
    ("it36l2i8s4e1c", "l2i6s4e1c", "1.0"),
    ("it36l2i8s4e1d", "l2i6s4e1d", "3.0"),
    ("it36l2i8s4e1e", "l2i6s4e1e", "10.0"),
]
# (corruption config name, what the row is) - only the winner's checkpoint ever exists.
COR_POINTS = [
    ("l2i6s4e1c", "chain 24 (E1 lambda=1.0 on row 45's 2,944-bit layout)"),
    ("l2i6s4e1d", "chain 24 (E1 lambda=3.0 on row 45's 2,944-bit layout)"),
    ("l2i6s4e1e", "chain 24 (E1 lambda=10.0 on row 45's 2,944-bit layout)"),
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


def sweep(stamp):
    """(ahead delta, imm delta, ahead ub95, imm ub95) or None if there is no usable verdict."""
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
    a, i = g.get("ahead") or {}, g.get("imm") or {}
    if a.get("delta") is None or i.get("delta") is None:
        return None
    return float(a["delta"]), float(i["delta"]), a.get("ub95"), i.get("ub95")


def gate_passed(stamp):
    s = sweep(stamp)
    if s is None or s[2] is None or s[3] is None:
        return None
    return bool(s[2] <= GATE_UB95 and s[3] <= GATE_UB95)


def pick_winner():
    ctl = sweep(CONTROL_STAMP)
    if ctl is None:
        log(f"no control verdict ({CONTROL_STAMP}) yet -- E1 has no measured baseline, so nothing runs")
        return None
    log(f"control {CONTROL_STAMP}: d ahead {ctl[0]:+.6f}, d imm {ctl[1]:+.6f}")
    winners = []
    for e1_stamp, cand, lam in COMBOS:
        s = sweep(e1_stamp)
        if s is None:
            log(f"  lambda {lam}: no verdict for {e1_stamp} -- skipped")
            continue
        better = s[0] < ctl[0] and s[1] < ctl[1]
        log(f"  lambda {lam}: d ahead {s[0]:+.6f}, d imm {s[1]:+.6f} -> "
            f"{'BEATS the control on both heads' if better else 'does not beat the control'}")
        if better:
            winners.append(((s[0] + s[1]) / 2.0, e1_stamp, cand, lam, s))
    if not winners:
        log("no lambda beat the penalty-free control on both heads -- E1 buys no margin here; "
            "nothing runs")
        return None
    winners.sort()
    agg, e1_stamp, cand, lam, s = winners[0]
    log(f"winner: lambda {lam} ({e1_stamp}, mean delta {agg:+.6f}) -> running {cand} at {STEPS} steps")
    return cand


def main():
    log("=== it36e1combo chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36e1combo chain end (without running anything) ===")
        return 1

    cand = pick_winner()
    if cand is not None:
        stamp = f"it36{cand}"
        epochs = max(2, round(STEPS / 62))
        log(f"point {cand}@{STEPS} -> tag {stamp}, ref {REF}, epochs {epochs}")
        rc = lib.qat_point(log, cand, stamp, STEPS, REF, epochs)
        if rc != 0:
            log(f"{cand}: point returned rc={rc}")

    # CORRUPTION PHASE (Andrew's standing order: never leave a corruption check pending).
    for name, note in COR_POINTS:
        out = COR / f"corruption_corrupt_{name}.json"
        if not (RUN / f"qat_it36{name}" / "qat_124.pth").exists():
            continue                      # a lambda that was not run has no checkpoint and no row
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS":
                    log(f"{name}: PASS already recorded -- skipping")
                    continue
            except (KeyError, ValueError):
                pass
        ok = gate_passed(f"it36{name}")
        if ok is False:
            log(f"{name}: fails the log-loss gate -- not a deploy candidate, no check needed")
            continue
        if ok is None:
            log(f"{name}: no usable sweep verdict yet -- skipping (a missing verdict is never a pass)")
            continue
        log(f"--- point {name} ({note}) passes the gate: running the corruption check")
        rc = lib.run(log, f"corrupt {name}",
                     [str(PY), "-u", str(DRIVER), "--tag", name, "--device", "cuda"],
                     COR / f"launcher_{name}.log")
        if rc != 0:
            log(f"{name}: driver rc={rc} (a FAIL or INCOMPLETE verdict is in its JSON)")

    log("=== it36e1combo chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
