"""Re-base the PTQ rows on the ANCHOR checkpoint (Andrew: "re-score PTQ runs ... if you have been
using weights from a different run").

Waits for the quant-tax evals of the anchor-redo checkpoints to finish, then scores the PTQ schemes
on `pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth` (the anchor baseline itself) so every row in
RESULTS.md sits on ONE checkpoint:

  l1i4      rank-1 int4 factors + int4 shifts        512 / 1,536 bits
  cn2       full-matrix int2 (both streams)        2,176 / 6,528 bits  (exactly at the bar)
  l1i8      rank-1 int8 factors + int8 shifts      1,024 / 3,072 bits
  c2_nl1i8  card full-matrix int2 + note rank-1 int8 2,176 / 3,072 bits (the QAT redo's config)

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36ptq_chain.py
"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

WAIT_FOR = ["adw_qc2_nl1i8_it36qatq.jsonl", "adw_qcn2_it36qatq.jsonl",
            "adw_qc2_nl1i8_it36ctlq.jsonl"]
MODEL = "pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth"
CANDIDATES = "l1i4,cn2,l1i8,c2_nl1i8"
STAMP = "it36ptq"


def main():
    t0 = time.time()
    while True:
        missing = [n for n in WAIT_FOR if not (RESULT / n).exists()]
        if not missing:
            break
        if (time.time() - t0) / 60 > 180:
            print(f"GAVE UP waiting for {missing}")
            return 1
        print(f"waiting for {missing}", flush=True)
        time.sleep(60)
    print("quant-tax evals finished; starting the PTQ re-base", flush=True)
    cmd = [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", MODEL,
           "--ref", "qnj_it36_124", "--candidates", CANDIDATES, "--shards", "2", "--solo",
           "1000000", "--stamp", STAMP, "--users-file", str(RUN / "roster750_seed20261119.json")]
    with (RUN / f"{STAMP}_chain.log").open("a", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"PTQ re-base done, rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
