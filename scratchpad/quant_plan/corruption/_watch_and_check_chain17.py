"""Run the corruption check for `l2i6s4` (RESULTS row 45) in the FIRST free GPU window.

WHY A WATCHER. `l2i6s4` PASSED the log-loss gate (2,944 bits - the new axis-1 frontier), and
Andrew's standing order is that a corruption check follows a passing row immediately. Chain 17
(`it36int6`) has no corruption phase of its own and is still running its last two points, and
`it36int6card` (chain 18) is ALREADY WAITING for chain 17's end marker, so there is no hand-off where
a normal job could slip in - a chain hands off inside a point with no idle gap, and both sides poll
once a minute.

THE MECHANISM (the `gpu_hold.py` pattern). Every waiter counts python processes whose command line
matches its BUSY_PAT, which for chain 18 includes `check_corruption`. So while a CPU-only process
whose command line carries that token is alive, chain 18 sees `busy > 0` and keeps waiting - even
after chain 17 has ended. This script keeps such a hold alive (renewing it before it can expire),
waits for chain 17's end marker, runs the check inside the held window, then releases the hold, which
lets chain 18 start. The hold only sleeps: it can never create a second GPU workload.

FAIL-SAFES. The hold self-expires (3600 s), so if this watcher dies the queue frees itself and chain
18 proceeds. The check is launched once and only once; after it returns, the hold is killed and this
script exits. If chain 17 has NOT ended and the hold has died, this script starts another hold, so the
queue can never open while the check is still pending.

Usage (detached, see detach.ps1):
  .venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\corruption\\_watch_and_check_chain17.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent           # scratchpad/quant_plan/corruption
RUN = HERE.parent                                # scratchpad/quant_plan
ROOT = HERE.parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CHAIN_LOG = RUN / "it36int6_chain.log"
CHAIN_END = "=== it36int6 chain end ==="
TAG = "l2i6s4"
CHECK = HERE / "check_corruption.py"
HOLD = HERE / "gpu_hold.py"
HOLD_SECS = 3600
LOG = HERE / f"watch_and_check_{TAG}.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def chain_ended() -> bool:
    return CHAIN_LOG.exists() and CHAIN_END in CHAIN_LOG.read_text(encoding="utf-8")


def start_hold():
    fh = (HERE / f"gpu_hold_{TAG}.log").open("a", encoding="utf-8")
    p = subprocess.Popen([str(PY), "-u", str(HOLD), "--tag", "check_corruption",
                          "--secs", str(HOLD_SECS)],
                         cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    log(f"queue hold started: pid={p.pid} tag=check_corruption secs={HOLD_SECS}")
    return p


def main() -> int:
    log(f"=== watcher start: waiting for '{CHAIN_END}' to run the {TAG} check ===")
    hold = start_hold()
    while not chain_ended():
        if hold.poll() is not None:
            log("hold exited before the chain ended -- renewing it")
            hold = start_hold()
        time.sleep(60)
    log("chain 17 logged its end marker")
    # Let chain 17's own processes leave the process table; the hold keeps chain 18 waiting.
    time.sleep(45)
    log(f"--- running the corruption check for {TAG} (the GPU is free, the hold is up)")
    t0 = time.time()
    with (HERE / f"launcher_{TAG}.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.run([str(PY), "-u", str(CHECK), "--tag", TAG, "--device", "cuda"],
                            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"driver rc={rc} in {(time.time() - t0) / 60:.1f} min")
    verdict = HERE / f"corruption_corrupt_{TAG}.json"
    if verdict.exists():
        log("verdict: " + verdict.read_text(encoding="utf-8").replace("\n", " ")[:400])
    else:
        log(f"NO verdict file at {verdict} -- read launcher_{TAG}.log")
    hold.kill()
    log("queue hold released -- the next chain may take the GPU")
    log("=== watcher end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
