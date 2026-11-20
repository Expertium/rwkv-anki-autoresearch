"""Third QAT screening slot on the iter36 anchor: candidate `l1i4` (rank-1 int4, both streams).

WHY. `l1i4` is 512 card bits / 1,536 note bits - FOUR times under the constraint-2 bar (2,176 /
6,528) and 2x cheaper than the currently compliant `c2_nl1i8` (2,176 / 3,072). Its untrained (PTQ)
cost is about +0.0009/+0.0014 on the old checkpoint, and QAT roughly removed that much again on the
`c2_nl1i8` config, so `l1i4` under QAT may pass the +0.0005 micro-bar AT A QUARTER OF THE BITS. It
also has a fused fast path in the deploy engine (0.903x rev/s = PASS on constraint 6).

QUEUE POSITION. Waits for `result/adw_qc2_nl1i8_it36qat496q.jsonl`, the last job of
`it36qat496_chain.py`, so only one GPU workload runs at a time.

  1. smoke   0.04 epochs (separate tag/prefix/folder)
  2. QAT     124 steps from rwkv_iter36_558.pth, candidate l1i4
  3. eval    the fp32 twin (`nj`) vs the anchor tag qnj_it36_124
  4. eval    the quantized deploy number (`l1i4`) vs that twin

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatl1i4_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatl1i4_chain.log"

WAIT_FOR = RESULT / "adw_qc2_nl1i8_it36qat496q.jsonl"
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qat496_chain.py|rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 300
ROSTER = RUN / "roster750_seed20261119.json"
CANDIDATE = "l1i4"
TAG = "it36qatl1i4"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def running(pattern):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -match 'python|rwkv-infer' -and "
         f"$_.CommandLine -match '{pattern}' }} | Measure-Object).Count"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def wait_for_gpu():
    t0 = time.time()
    while True:
        done = WAIT_FOR.exists()
        busy = running(BUSY_PAT)
        if busy == 0 and done:
            log(f"GPU free: {WAIT_FOR.name} exists, 0 sweep/train processes")
            return True
        if (time.time() - t0) / 60 > WAIT_MAX_MIN:
            log(f"GAVE UP after {WAIT_MAX_MIN} min (done={done}, busy={busy})")
            return False
        time.sleep(60)
        log(f"waiting: {WAIT_FOR.name} exists={done}, sweep/train processes={busy}")


def run(tag, cmd, logfile):
    log(f"START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def main():
    log("=== it36qatl1i4 chain start ===")
    if not wait_for_gpu():
        return

    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--train-users", "1", "100",
            "--max-train-len", "66000", "--start-folder", "pretrain/rwkv/opt_iter36",
            "--start-name", "rwkv_iter36_558"]

    rc = run("smoke", base + ["--epochs", "0.04", "--max-steps", "6", "--candidate", CANDIDATE,
                              "--tag", f"{TAG}smoke"], RUN / f"{TAG}smoke_launcher.log")
    if rc != 0:
        log("smoke FAILED -- stopping")
        return
    rc = run("qat-l1i4", base + ["--epochs", "2", "--max-steps", "124", "--candidate", CANDIDATE,
                                 "--prefix", "qat", "--tag", TAG], RUN / f"{TAG}_launcher.log")
    if rc != 0:
        log("QAT FAILED -- stopping")
        return

    ckpt = RUN / f"qat_{TAG}" / "qat_124.pth"
    if not ckpt.exists():
        log(f"checkpoint missing, cannot evaluate: {ckpt}")
        return
    run("eval-fp32", [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                      "--ref", "qnj_it36_124", "--candidates", "nj", "--shards", "2",
                      "--solo", "1000000", "--stamp", TAG, "--users-file", str(ROSTER)],
        RUN / f"eval_{TAG}_launcher.log")
    run("eval-quant", [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                       "--ref", f"qnj_{TAG}", "--candidates", CANDIDATE, "--shards", "2",
                       "--solo", "1000000", "--stamp", f"{TAG}q", "--users-file", str(ROSTER)],
        RUN / f"eval_{TAG}q_launcher.log")
    log("=== it36qatl1i4 chain end ===")


if __name__ == "__main__":
    sys.exit(main())
