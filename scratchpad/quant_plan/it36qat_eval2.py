"""Measure the DEPLOY tax of the anchor-redo checkpoints (the numbers the deliverable needs).

The two training runs are already done and both were scored UNQUANTIZED (candidate `nj`):
  * `qnj_it36ctl` = the fp32 control   (budget-matched base: 124-step decay, no quantization)
  * `qnj_it36qat` = the QAT run        (same 124 steps, card full-matrix int2 + note rank-1 int8)

This script adds the missing half: the SAME checkpoints evaluated WITH the quantization on (the
`c2_nl1i8` candidate, i.e. the deploy simulation), plus full-matrix `cn2` on the QAT checkpoint
(both streams int2, exactly AT the 2,176 / 6,528 bit bar - the simplest scheme there is).

Pairings to read afterwards:
  * `qnl1i8? on qat` vs `qnj_it36qat`  -> the QUANT COST on a QAT-trained model (the arm's question)
  * `qnl1i8? on ctl` vs `qnj_it36ctl`  -> the PTQ COST at the same base (why QAT is needed)
  * `qnl1i8? on qat` vs `qnj_it36ctl`  -> the FULL TAX vs the fp32 twin of the same budget
  * `cn2? on qat`    vs `qnj_it36qat`  -> does QAT dissolve the simplest at-the-bar scheme too?

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qat_eval2.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

JOBS = [
    # (stamp, checkpoint, candidates, ref tag)
    ("it36qatq", RUN / "qat_it36qat" / "qat_124.pth", "c2_nl1i8,cn2", "qnj_it36qat"),
    ("it36ctlq", RUN / "qat_it36ctl" / "ctl_124.pth", "c2_nl1i8", "qnj_it36ctl"),
]

for stamp, model, cands, ref in JOBS:
    print(f"\n=== {stamp}: {model.name} candidates={cands} ref={ref}", flush=True)
    cmd = [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(model), "--ref", ref,
           "--candidates", cands, "--shards", "2", "--solo", "1000000", "--stamp", stamp,
           "--users-file", str(RUN / "roster750_seed20261119.json")]
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"=== {stamp} done, rc={rc}", flush=True)
print("\nQUANT-TAX EVALS DONE", flush=True)
