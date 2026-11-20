"""Score anchor candidates (unquantized) on the 750-user roster, one after another.

Andrew's rule for the anchor ("the best run"): it must have a checkpoint BEFORE its decay phase
(so the decay can be redone with QAT) and AFTER it (for the baseline score), and among such runs
the one with the LOWEST log loss on both heads wins.

CORRECTED MAPPING (2026-11-19 22:20, from the artifact arithmetic -- see NOTES.md "Anchor
investigation"): the phase that RESTARTS the step counter is the DECAY. qat_log #40 says the full
iter36 run was "558 WS + 124 decay", and the decay runs (iter39/43 = 124, iter44 = 248, iter45 =
496, iter46 = 248) all restart at 0, so:

    rwkv_iter36_558.pth = iter36 WS-final   -> the PRE-decay checkpoint (the QAT start point)
    rwkv_iter36_124.pth = iter36 post-decay -> the ANCHOR BASELINE (the [1,4,3,3,3] champion)

This is the reverse of the earlier note in NOTES.md/RESULTS.md, which had the two swapped.

This scores the AFTER checkpoints with the arm's own harness (`quant_sweep.py` with candidate `nj` =
no quantization, eager path), so every candidate gets a paired per-user delta against the existing
baseline tag `qbl_incumbent_a04`. Each job is 750 users in two shards plus the solo phase (~35 min).
Already-finished jobs are skipped, so the script is safe to re-run.

Usage (run detached; it chains the jobs itself):
  .venv/Scripts/python.exe scratchpad/quant_plan/score_anchors.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"

# stamp -> (checkpoint, what it is / where its decay started)
JOBS = [
    ("it36_124", ("pretrain/rwkv/opt_iter36/rwkv_iter36_124.pth",
                  "iter36 POST-decay = the [1,4,3,3,3] champion (WS 558 + 124-step decay); "
                  "THE ANCHOR BASELINE")),
    ("it36_558", ("pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth",
                  "iter36 WS-FINAL = the PRE-decay checkpoint every QAT run started from")),
    ("it45_496", ("pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth",
                  "iter45 = 16-epoch decay-QAT from iter36 WS-final (qat_log #45, best-imm champion)")),
    ("it46_248", ("pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth",
                  "iter46 = last old-regime QAT run; the checkpoint the adaptive run started from")),
    ("it44_248", ("pretrain/rwkv/opt_qat44/rwkv_iter44_248.pth",
                  "iter44 = 8-epoch decay-QAT from iter36 WS-final (qat_log #44)")),
]

for stamp, (model, why) in JOBS:
    done = (RESULT / f"adw_qnj_{stamp}.jsonl").exists() and (RESULT / f"adwP_qnj_{stamp}.jsonl").exists()
    if done:
        print(f"=== skipping {stamp} (canonical result already exists)", flush=True)
        continue
    print(f"\n=== scoring {stamp}: {why}", flush=True)
    cmd = [str(PY), str(RUN / "quant_sweep.py"), "--model", model, "--ref", "qbl_incumbent_a04",
           "--candidates", "nj", "--shards", "2", "--solo", "1000000", "--stamp", stamp,
           "--users-file", str(RUN / "roster750_seed20261119.json")]
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"=== {stamp} done, rc={rc}", flush=True)
print("\nALL ANCHOR SCORES DONE", flush=True)
