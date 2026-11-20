"""Second QAT screening run on the iter36 anchor: candidate `l1i8` (rank-1 int8, both streams).

WHY. The anchor redo (it36qat_chain.py) screened ONE QAT config, `c2_nl1i8` (card full-matrix int2 +
note rank-1 int8, 2,176/3,072 bits = exactly at / half the bar). The PTQ sweep says the rank-1 family
spends its bits far better than the full-matrix family (iter 3 = rank-1 int8 = best untrained scheme,
1,024/3,072 bits, 2x under the bar), so the natural second QAT screening slot is rank-1 int8 itself:
it is the best PTQ scheme AND it has 2x the bit headroom. The 124-step bed costs ~4 min of QAT, so the
screening is nearly free compared with the 12 h of a real decay.

ORDER (one GPU workload at a time):
  1. wait for the PTQ re-base (`result/adw_qc2_nl1i8_it36ptq.jsonl`, the ptq chain's last job) and
     for the deploy-tax evals to drain, then take the GPU
  2. smoke  -- 0.04 epochs of the exact path, candidate l1i8, separate tag/prefix/folder
  3. QAT    -- 124 steps from rwkv_iter36_558.pth (the pre-decay WS-final), candidate l1i8
  4. eval   -- unquantized (`nj`) vs tag qnj_it36_124  -> the fp32 twin / QAT-training effect
  5. eval   -- quantized (`l1i8,c2_nl1i8`) vs that twin -> the DEPLOY number for constraint 3
  6. export -- the deploy artifact for the new checkpoint
  7. speed  -- rust_speed.py on the IDLE machine (constraint 6) for the configs of this arm

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatl1i8_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
ART = RUN / "rust_artifacts"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatl1i8_chain.log"

# the ptq chain's last job writes this; the deploy-tax evals (eval2) all finish before it does
RESULT_DIR = ROOT / "result"
WAIT_FOR = RESULT_DIR / "adw_qc2_nl1i8_it36ptq.jsonl"
SWEEP_PAT = "quant_sweep.py"
WAIT_MAX_MIN = 240

TAG = "it36qatl1i8"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def running(pattern):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -match 'python' -and "
         f"$_.CommandLine -match '{pattern}' }} | Measure-Object).Count"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def wait_for_gpu():
    """Return once the ptq chain is done and no roster sweep is on the GPU.

    Primary rule: WAIT_FOR (the ptq chain's last job) exists and 0 sweep processes. Safety net in
    case that job dies: any `adw_q*_it36ptq.jsonl` exists and the sweep has been idle for 15 min --
    long enough that the gap between two candidates (seconds) cannot be mistaken for the end.
    """
    t0 = time.time()
    idle_since = None
    while True:
        done = WAIT_FOR.exists()
        any_ptq = next(RESULT_DIR.glob("adw_q*_it36ptq.jsonl"), None) is not None
        busy = running(SWEEP_PAT)
        if busy == 0:
            idle_since = idle_since or time.time()
        else:
            idle_since = None
        idle_min = (time.time() - idle_since) / 60 if idle_since else 0.0
        if busy == 0 and (done or (any_ptq and idle_min >= 15)):
            log(f"GPU free: {WAIT_FOR.name} exists={done}, sweep idle {idle_min:.0f} min")
            return True
        if (time.time() - t0) / 60 > WAIT_MAX_MIN:
            log(f"GAVE UP after {WAIT_MAX_MIN} min (done={done}, any_ptq={any_ptq}, busy={busy})")
            return False
        time.sleep(60)
        log(f"waiting: {WAIT_FOR.name} exists={done}, any_ptq={any_ptq}, sweep processes={busy}")


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
    log("=== it36qatl1i8 chain start ===")
    if not wait_for_gpu():
        return

    decay = [str(PY), "-u", str(RUN / "qat_decay.py"),
             "--epochs", "2", "--max-steps", "124", "--train-users", "1", "100",
             "--max-train-len", "66000", "--start-folder", "pretrain/rwkv/opt_iter36",
             "--start-name", "rwkv_iter36_558"]

    smoke = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", "l1i8",
             "--epochs", "0.04", "--train-users", "1", "100", "--max-train-len", "66000",
             "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558",
             "--tag", f"{TAG}smoke"]
    rc = run("smoke", smoke, RUN / f"{TAG}smoke_launcher.log")
    if rc != 0:
        log("smoke FAILED -- stopping the chain before the full run")
        return

    rc = run("qat-l1i8", decay + ["--candidate", "l1i8", "--prefix", "qat", "--tag", TAG],
             RUN / f"{TAG}_launcher.log")
    if rc != 0:
        log("QAT FAILED -- stopping the chain")
        return

    ckpt = RUN / f"qat_{TAG}" / "qat_124.pth"
    if not ckpt.exists():
        log(f"checkpoint missing, cannot evaluate: {ckpt}")
        return

    # the fp32 twin: same QAT-trained weights, evaluated with NO quantization
    run("eval-fp32", [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                      "--ref", "qnj_it36_124", "--candidates", "nj", "--shards", "2",
                      "--solo", "1000000", "--stamp", TAG,
                      "--users-file", str(RUN / "roster750_seed20261119.json")],
        RUN / f"eval_{TAG}_launcher.log")

    # the DEPLOY number: the same weights with quantization ON, measured against the twin above
    run("eval-quant", [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                       "--ref", f"qnj_{TAG}", "--candidates", "l1i8,c2_nl1i8", "--shards", "2",
                       "--solo", "1000000", "--stamp", f"{TAG}q",
                       "--users-file", str(RUN / "roster750_seed20261119.json")],
        RUN / f"eval_{TAG}q_launcher.log")

    # deploy artifact + constraint 6 speed, both on the now-idle machine
    sft = ART / f"{TAG}_124.safetensors"
    run("export", [str(PY), "-u", str(RUN / "export_weights_only.py"), str(ckpt.relative_to(ROOT)),
                   str(ART), sft.name], RUN / f"export_{TAG}.log")
    if sft.exists():
        run("speed", [str(PY), "-u", str(RUN / "rust_speed.py"), "--weights", str(sft),
                      "--configs", "c2_nl1i8,cn2,l1i8,l1i4", "--reps", "2", "--tag", TAG],
            RUN / f"speed_{TAG}.log")
    else:
        log("export produced no artifact -- skipping the speed run")
    log("=== it36qatl1i8 chain end ===")


if __name__ == "__main__":
    sys.exit(main())
