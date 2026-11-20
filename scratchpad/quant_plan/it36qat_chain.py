"""Anchor redo chain (Andrew 2026-11-19: anchor = iter36).

Runs, in order, on ONE GPU at a time:

  1. wait for the anchor baseline eval (`result/adw_qnj_it36_124.jsonl`) to appear, then take the GPU
  2. smoke   -- 0.04 epochs (6 steps) of the exact experiment path, separate tag/prefix/folder
  3. control -- the 124-step decay from iter36 WS-final with NO quantization (`--no-quant`)
  4. QAT     -- the same 124 steps with card full-matrix int2 + note rank-1 int8 (candidate c2_nl1i8)
  5. evals   -- both new checkpoints on the 750-user roster, unquantized harness (`nj`), so each gets
                a paired per-user delta vs the anchor baseline tag qnj_it36_124
  6. diags   -- then re-launch score_anchors.py (it skips finished jobs) for the remaining anchors

Why 124 steps: iter36's own decay was the "124-step decay" of qat_log #39/#40 ("558 WS + 124 decay");
the decay phase restarts its step counter, so rwkv_iter36_558.pth is the WS-final (the pre-decay
checkpoint, = the QAT start) and rwkv_iter36_124.pth is the post-decay champion (= the baseline).

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qat_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ANCHOR_RESULT = ROOT / "result" / "adw_qnj_it36_124.jsonl"
LOG = RUN / "it36qat_chain.log"

KILL_PAT = "score_anchors.py|quant_sweep.py|eval_sharded.py|rwkv.get_result|rwkv.train_rwkv"
WAIT_MAX_MIN = 90


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


def wait_for_anchor_eval():
    t0 = time.time()
    while not ANCHOR_RESULT.exists():
        if (time.time() - t0) / 60 > WAIT_MAX_MIN:
            log(f"GAVE UP waiting for {ANCHOR_RESULT.name} after {WAIT_MAX_MIN} min")
            return False
        time.sleep(60)
        log("waiting for the anchor baseline eval ...")
    log(f"anchor baseline eval finished ({ANCHOR_RESULT.name} exists)")
    return True


def free_gpu(tag):
    """Stop the eval chain (it would start the next diagnostic job) and wait for the GPU to drain."""
    n = running(KILL_PAT)
    log(f"stopping the eval chain ({n} matching process(es)) before {tag}")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "$pat='" + KILL_PAT + "'; Get-CimInstance Win32_Process | "
                    "Where-Object { $_.Name -match 'python' -and $_.CommandLine -match $pat } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                   capture_output=True, text=True)
    for _ in range(30):
        time.sleep(10)
        if running(KILL_PAT) == 0:
            break
    log(f"GPU clear ({running(KILL_PAT)} matching processes left)")


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
    log("=== it36qat chain start ===")
    if not wait_for_anchor_eval():
        return
    free_gpu("the training runs")

    decay = [str(PY), "-u", str(RUN / "qat_decay.py"),
             "--epochs", "2", "--max-steps", "124", "--train-users", "1", "100",
             "--max-train-len", "66000", "--start-folder", "pretrain/rwkv/opt_iter36",
             "--start-name", "rwkv_iter36_558"]

    # 2. smoke (0.04 epochs = 6 steps) -- the AGENTS.md requirement for a non-HP experiment
    smoke = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", "c2_nl1i8",
             "--epochs", "0.04", "--train-users", "1", "100", "--max-train-len", "66000",
             "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558",
             "--tag", "it36qatsmoke"]
    rc = run("smoke", smoke, RUN / "it36qatsmoke_launcher.log")
    if rc != 0:
        log("smoke FAILED -- stopping the chain before the full runs")
        return

    # 3. control (fp32) then 4. QAT, both exactly 124 steps
    rc = run("control", decay + ["--candidate", "c2_nl1i8", "--no-quant", "--prefix", "ctl",
                                 "--tag", "it36ctl"], RUN / "it36ctl_launcher.log")
    if rc != 0:
        log("control FAILED -- stopping the chain")
        return
    rc = run("qat", decay + ["--candidate", "c2_nl1i8", "--prefix", "qat", "--tag", "it36qat"],
             RUN / "it36qat_launcher.log")
    if rc != 0:
        log("QAT FAILED -- stopping the chain")
        return

    # 5. evals: the unquantized harness on both new checkpoints, paired against the anchor baseline
    for tag, ckpt in (("it36ctl", RUN / "qat_it36ctl" / "ctl_124.pth"),
                      ("it36qat", RUN / "qat_it36qat" / "qat_124.pth")):
        if not ckpt.exists():
            log(f"checkpoint missing, cannot eval: {ckpt}")
            continue
        cmd = [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
               "--ref", "qnj_it36_124", "--candidates", "nj", "--shards", "2", "--solo", "1000000",
               "--stamp", tag, "--users-file", str(RUN / "roster750_seed20261119.json")]
        run(f"eval {tag}", cmd, RUN / f"eval_{tag}_launcher.log")

    # 6. the remaining anchor diagnostics (finished jobs are skipped)
    run("diag anchors", [str(PY), "-u", str(RUN / "score_anchors.py")], RUN / "score_anchors.log")
    log("=== it36qat chain end ===")


if __name__ == "__main__":
    sys.exit(main())
