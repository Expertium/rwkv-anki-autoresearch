"""Does a LONGER QAT decay close the remaining quantization tax on the iter36 anchor?

WHY. The 124-step anchor redo passes Andrew's log-loss bar against the anchor champion (row 7 of
RESULTS.md: 0.335038/0.307253 vs 0.342813/0.314272), but measured against its own budget-matched
fp32 twin it still costs +0.001236 ahead / +0.001042 imm (ub95 +0.001375/+0.001205) -- 2.4-2.8x over
the +0.0005 ceiling. History says this tax shrinks with the length of the QAT decay (qat_log #39-#45:
the longer decay-QAT runs lost less to the quantizer), so this chain measures the SAME candidate at
iter45's decay length (496 steps = 16 groups of 31 of the old config) together with its own fp32 twin.
The pair is what keeps the two effects apart: the longer decay moves the model, the quantizer costs
what the twin comparison says.

QUEUE POSITION. It waits for the second QAT screening chain (`it36qatl1i8_chain.py`) to publish
`result/adw_qc2_nl1i8_it36qatl1i8q.jsonl`, so nothing overlaps on the GPU.

  smoke   0.04 epochs, candidate c2_nl1i8 (separate tag/prefix/folder)
  control 496 steps, --no-quant (the fp32 twin)
  QAT     496 steps, candidate c2_nl1i8
  evals   fp32 twins of both vs tag qnj_it36_124, then the quantized deploy number of each vs its
          own twin

Scope note: 496 steps is 4x iter36's own 124-step decay, i.e. a DIAGNOSTIC of how the tax scales with
QAT length, not the faithful replay (that is `it36qat`, already done). It is cheap (~15 min of QAT,
~40 min of fp32 control on this bed) and it is the only lever left that costs no bits.

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qat496_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qat496_chain.log"

WAIT_FOR = RESULT / "adw_qc2_nl1i8_it36qatl1i8q.jsonl"
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatl1i8_chain.py|rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 240
ROSTER = RUN / "roster750_seed20261119.json"
CANDIDATE = "c2_nl1i8"
STEPS = 496


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
    t0, idle_since = time.time(), None
    while True:
        done = WAIT_FOR.exists()
        busy = running(BUSY_PAT)
        idle_since = (idle_since or time.time()) if busy == 0 else None
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
    log("=== it36qat496 chain start ===")
    if not wait_for_gpu():
        return

    common = [str(PY), "-u", str(RUN / "qat_decay.py"),
              "--train-users", "1", "100", "--max-train-len", "66000",
              "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558"]
    qat = common + ["--epochs", "8", "--max-steps", str(STEPS), "--candidate", CANDIDATE,
                    "--prefix", "qat"]
    ctl = common + ["--epochs", "8", "--max-steps", str(STEPS), "--candidate", CANDIDATE,
                    "--no-quant", "--prefix", "ctl"]

    smoke = ([str(PY), "-u", str(RUN / "qat_decay.py"), "--train-users", "1", "100",
              "--max-train-len", "66000", "--start-folder", "pretrain/rwkv/opt_iter36",
              "--start-name", "rwkv_iter36_558", "--epochs", "0.04", "--max-steps", "6",
              "--candidate", CANDIDATE, "--tag", "it36qat496smoke"])
    rc = run("smoke", smoke, RUN / "it36qat496smoke_launcher.log")
    if rc != 0:
        log("smoke FAILED -- stopping")
        return

    rc = run("control496", ctl + ["--tag", "it36ctl496"], RUN / "it36ctl496_launcher.log")
    if rc != 0:
        log("control496 FAILED -- stopping")
        return
    rc = run("qat496", qat + ["--tag", "it36qat496"], RUN / "it36qat496_launcher.log")
    if rc != 0:
        log("qat496 FAILED -- stopping")
        return

    both = [("it36ctl496", RUN / "qat_it36ctl496" / "ctl_496.pth"),
            ("it36qat496", RUN / "qat_it36qat496" / "qat_496.pth")]
    for tag, ckpt in both:
        if not ckpt.exists():
            log(f"checkpoint missing, cannot evaluate: {ckpt}")
            continue
        run(f"eval-fp32 {tag}",
            [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
             "--ref", "qnj_it36_124", "--candidates", "nj", "--shards", "2", "--solo", "1000000",
             "--stamp", tag, "--users-file", str(ROSTER)],
            RUN / f"eval_{tag}_launcher.log")
    for tag, ckpt in both:
        if not ckpt.exists():
            continue
        run(f"eval-quant {tag}",
            [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
             "--ref", f"qnj_{tag}", "--candidates", CANDIDATE, "--shards", "2", "--solo", "1000000",
             "--stamp", f"{tag}q", "--users-file", str(ROSTER)],
            RUN / f"eval_{tag}q_launcher.log")
    log("=== it36qat496 chain end ===")


if __name__ == "__main__":
    sys.exit(main())
