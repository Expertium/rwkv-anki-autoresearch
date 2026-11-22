"""Run the corruption checks for chain 18's two points in the FIRST free GPU window.

WHY A WATCHER. `it36int6card` (chain 18) has NO corruption phase of its own - it runs only its two
QAT points (`l2i6c` at 1,024/2,304 = 3,328 bits, then `l2i6n` at 1,280/1,920 = 3,200 bits). Both sit
above row 45's 2,944 bits, so they are diagnostic rows (they say WHICH stream owns the int6 tax), but
Andrew's standing order is that a row that passes the log-loss gate gets its finiteness check
immediately. The problem is the hand-off: chain 18 hands the GPU straight to `it36fixedbasis` (chain
19) with no idle gap - both sides poll once a minute - so a normal job cannot slip in.

THE MECHANISM (`gpu_hold.py`). Every waiter counts python/rwkv-infer processes whose command line
matches its BUSY_PAT, and every `it36*_chain.py` BUSY_PAT contains `check_corruption`. So while a
CPU-only process whose command line carries that token is alive, chain 19 sees `busy > 0` and keeps
waiting - even after chain 18 has ended. This script keeps such a hold alive (renewing it before it can
expire), waits for chain 18's end marker, runs one check per POINT THAT PASSED THE LOG-LOSS GATE, then
releases the hold, which lets chain 19 start. The hold only sleeps: it can never create a second GPU
workload.

FAIL-SAFES. The hold self-expires (3,600 s), so if this watcher dies the queue frees itself and chain
19 proceeds. Each check is launched once and only once. If chain 18 has NOT ended and the hold has
died, this script starts another hold, so the queue can never open while a check is still pending.

Usage (detached, see detach.ps1):
  .venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\corruption\\_watch_and_check_chain18.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent           # scratchpad/quant_plan/corruption
RUN = HERE.parent                                # scratchpad/quant_plan
ROOT = HERE.parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CHAIN_LOG = RUN / "it36int6card_chain.log"
CHAIN_END = "=== it36int6card chain end ==="
CHECK = HERE / "check_corruption.py"
HOLD = HERE / "gpu_hold.py"
HOLD_SECS = 3600
GATE_UB95 = 0.0005
# (config name, sweep stamp) - the two chain-18 points, in run order
POINTS = [("l2i6c", "it36l2i6c"), ("l2i6n", "it36l2i6n")]
LOG = HERE / "watch_and_check_chain18.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def chain_ended() -> bool:
    return CHAIN_LOG.exists() and CHAIN_END in CHAIN_LOG.read_text(encoding="utf-8")


def start_hold():
    fh = (HERE / "gpu_hold_chain18.log").open("a", encoding="utf-8")
    p = subprocess.Popen([str(PY), "-u", str(HOLD), "--tag", "check_corruption",
                          "--secs", str(HOLD_SECS)],
                         cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    log(f"queue hold started: pid={p.pid} tag=check_corruption secs={HOLD_SECS}")
    return p


def gate_pass(stamp: str):
    """True/False/None - None means the sweep has not landed, so nothing can be decided."""
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        return None
    g = rows[0].get("gate", {})
    ahead, imm = g.get("ahead", {}), g.get("imm", {})
    if "ub95" not in ahead or "ub95" not in imm:
        return None
    ok = bool(rows[0].get("bits_ok")) and ahead["ub95"] <= GATE_UB95 and imm["ub95"] <= GATE_UB95
    log(f"  {stamp}: bits_ok={rows[0].get('bits_ok')} ub95 ahead={ahead['ub95']:+.6f} "
        f"imm={imm['ub95']:+.6f} -> gate {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    log(f"=== watcher start: waiting for '{CHAIN_END}' to run the {[p[0] for p in POINTS]} checks ===")
    hold = start_hold()
    while not chain_ended():
        if hold.poll() is not None:
            log("hold exited before the chain ended -- renewing it")
            hold = start_hold()
        time.sleep(60)
    log("chain 18 logged its end marker")
    # Let chain 18's own processes leave the process table; the hold keeps chain 19 waiting.
    time.sleep(45)
    ran = []
    for name, stamp in POINTS:
        verdict_file = HERE / f"corruption_corrupt_{name}.json"
        if verdict_file.exists():
            log(f"{name}: verdict already exists ({verdict_file.name}) -- not re-running it")
            continue
        ok = gate_pass(stamp)
        if ok is None:
            log(f"{name}: no usable sweep at quant_sweep_{stamp}.json -- SKIPPING (cannot decide)")
            continue
        if not ok:
            log(f"{name}: fails the log-loss gate -- not a deploy candidate, no check needed")
            continue
        log(f"--- running the corruption check for {name} (the GPU is free, the hold is up)")
        t0 = time.time()
        with (HERE / f"launcher_{name}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.run([str(PY), "-u", str(CHECK), "--tag", name, "--device", "cuda"],
                                cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
        log(f"driver rc={rc} in {(time.time() - t0) / 60:.1f} min")
        if verdict_file.exists():
            log("verdict: " + verdict_file.read_text(encoding="utf-8").replace("\n", " ")[:400])
        else:
            log(f"NO verdict file at {verdict_file} -- read launcher_{name}.log")
        ran.append(name)
    hold.kill()
    log(f"queue hold released -- the next chain may take the GPU (checks run: {ran or 'none'})")
    log("=== watcher end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
