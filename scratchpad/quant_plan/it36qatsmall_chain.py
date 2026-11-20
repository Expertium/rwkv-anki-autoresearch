"""Smaller-state screening on the iter36 anchor (Andrew 2026-11-20 22:23).

Andrew's new reading of the gate: the comparison must be **"whatever run that is exactly identical
EXCEPT for quantization"**, i.e. the fp32 control trained with the same recipe, data and step budget
(`qnj_it36ctl`, 124 steps, 0.333802/0.306211) -- not the iter36 champion, which differs in data too.
His priority order for further improvement: (1) SMALLER state size, (2) lower log loss against that
identical-but-unquantized run, (3) speed (low priority).

Best row so far: `l1i8` (rank-1 int8 on both streams) = 1,024 card bits + 3,072 note bits, full tax
+0.000503 ahead / +0.001009 imm against the control. This chain walks DOWN the bit ladder, because
that is the axis he ranks first. Every point is one QAT run plus one quantized eval against the same
control, so the points form a bits-vs-tax curve.

  ladder (124 steps, ref = qnj_it36ctl):
    l1i4     rank-1 int4 factors + int4 shifts   512 / 1,536 bits  (half of l1i8)
    l1i4s2   rank-1 int4 factors + int2 shifts   384 / 1,152 bits
    l1i2     rank-1 int2 factors + int2 shifts   256 /   768 bits  (quarter of l1i8)
  axis 2 (same bits as the best row, longer QAT decay):
    l1i8 at 496 steps, ref = the 496-step fp32 control `ctl_496.pth` (trained 2026-11-20 10:21 with
    the same recipe and step cap, never evaluated -- job 0 scores it).

The 496-step twin is what makes the l1i8-at-496 point gate-comparable: every point is scored against
its OWN identical-but-unquantized run, never against a different budget.

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatsmall_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatsmall_chain.log"
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
CTL_496 = "qnj_it36ctl496"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(tag, cmd, logfile):
    log(f"START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def eval_checkpoint(label, ckpt, ref, cands, stamp):
    """Skip if the merged result already exists (relaunch-safe)."""
    tag = f"q{cands.split(',')[-1]}_{stamp}"
    if (RESULT / f"adw_{tag}.jsonl").exists():
        log(f"{label}: {tag} already exists -- skipping")
        return 0
    return run(label, [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                       "--ref", ref, "--candidates", cands, "--shards", "2", "--solo", "1000000",
                       "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def qat_point(cand, steps, epochs, stamp, ref):
    """0.04-epoch smoke -> QAT -> quantized eval against the identical unquantized run."""
    if (RESULT / f"adw_q{cand}_{stamp}.jsonl").exists():
        log(f"{cand}@{steps}: result exists -- skipping the whole point")
        return 0
    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", cand,
            "--train-users", "1", "100", "--max-train-len", "66000",
            "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558"]
    rc = run(f"smoke {cand}@{steps}",
             base + ["--epochs", "0.04", "--max-steps", "6", "--tag", f"{stamp}smoke"],
             RUN / f"{stamp}smoke_launcher.log")
    if rc != 0:
        log(f"smoke FAILED for {cand}@{steps} -- skipping this point")
        return rc
    rc = run(f"qat {cand}@{steps}",
             base + ["--epochs", str(epochs), "--max-steps", str(steps), "--prefix", "qat",
                     "--tag", stamp], RUN / f"{stamp}_launcher.log")
    if rc != 0:
        log(f"QAT FAILED for {cand}@{steps} -- skipping this point")
        return rc
    ckpt = RUN / f"qat_{stamp}" / f"qat_{steps}.pth"
    if not ckpt.exists():
        log(f"checkpoint missing: {ckpt}")
        return 1
    return eval_checkpoint(f"eval {cand}@{steps}", ckpt, ref, cand, stamp)


def main():
    log("=== it36qatsmall chain start ===")

    # job 0: the 496-step fp32 control, the identical-but-unquantized twin for the axis-2 point
    ctl496 = RUN / "qat_it36ctl496" / "ctl_496.pth"
    if ctl496.exists():
        eval_checkpoint("eval ctl496 (fp32 twin)", ctl496, "qnj_it36_124", "nj", "it36ctl496")
    else:
        log(f"missing {ctl496} -- the 496-step twin cannot be scored")

    # axis 1: the bit ladder, all against the SAME 124-step fp32 control
    for cand, stamp in (("l1i4", "it36small4"), ("l1i4s2", "it36smalls2"),
                        ("l1i2", "it36small2")):
        rc = qat_point(cand, 124, 2, stamp, CTL_124)
        if rc != 0:
            log(f"point {cand} failed (rc={rc}); continuing with the next one")

    # axis 2: the current best allocation at iter45's 496-step decay length
    qat_point("l1i8", 496, 8, "it36l1i8496", CTL_496)

    log("=== it36qatsmall chain end ===")


if __name__ == "__main__":
    sys.exit(main())