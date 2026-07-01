"""Correctness gate for a CPU-speed candidate: run the champion binary and the candidate binary
on the SAME user (run_user mode writes reference/rust_pred_<user>.json) and report the max abs
diff of pred_imm + pred_ahead. A pure-perf change should be ~0 (bit-identical); target-cpu=native
(FMA) may shift by ~1e-6. Accept-on-correctness threshold = 1e-5 (logloss impact negligible).

Usage: python check_parity.py <champion.exe> <candidate.exe> <weights.safetensors> [user]
Prints: PARITY_MAXDIFF <value>   and PASS/FAIL vs 1e-5.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("C:/Users/Andrew/rwkv-anki-autoresearch")
champ, cand, weights = sys.argv[1], sys.argv[2], sys.argv[3]
user = sys.argv[4] if len(sys.argv) > 4 else "107"
pred_path = ROOT / "reference" / f"rust_pred_{user}.json"


def run(binpath):
    env = {**os.environ, "RWKV_WEIGHTS": weights, "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    subprocess.run([binpath, user], cwd=str(ROOT), env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(pred_path))


a = run(champ)
b = run(cand)
md_imm = max(abs(x - y) for x, y in zip(a["pred_imm"], b["pred_imm"]))
# pred_ahead is Vec<Option<f32>> -> None for first-sight cards; compare where both present
pa, pb = a["pred_ahead"], b["pred_ahead"]
md_ah = 0.0
for x, y in zip(pa, pb):
    if x is not None and y is not None:
        md_ah = max(md_ah, abs(x - y))
md = max(md_imm, md_ah)
print(f"PARITY_MAXDIFF {md:.3e}  (imm {md_imm:.3e}, ahead {md_ah:.3e}, n={len(a['pred_imm'])})")
print("PASS (<1e-5)" if md < 1e-5 else "FAIL (>=1e-5)")
