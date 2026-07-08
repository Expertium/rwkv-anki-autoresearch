"""Champion-run finisher: sanity-check the full-eval result jsonls, compute by-user mean LogLoss
for both modes, then promote via optimization/promote_champion_5k.py (carrying ckpt + the run's
OWN learned codebooks, per the 2026-07-08 learnable-cb doctrine).

Usage:
  python scratchpad/champ5k_finish.py <name> <ws_trace> <ahead_jsonl> <imm_jsonl> \
      <ckpt_folder> <ckpt_prefix> <cb_wkv_txt> <cb_shift_txt> <expect_n>
The champion ckpt = max-step {ckpt_folder}/{ckpt_prefix}_<step>.pth (excl optim/ema) -- numeric
max, not name sort. Exits 1 (loud) if either jsonl has != expect_n users or any file is missing.
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
name, trace, fa, fi, ckpt_folder, ckpt_prefix, cb_wkv, cb_shift, expect_n = sys.argv[1:10]
expect_n = int(expect_n)

cands = []
for p in glob.glob(f"{ckpt_folder}/{ckpt_prefix}_*.pth"):
    b = os.path.basename(p)
    if "optim" in b or "ema" in b:
        continue
    m = re.match(rf"{re.escape(ckpt_prefix)}_(\d+)\.pth$", b)
    if m:
        cands.append((int(m.group(1)), p.replace("\\", "/")))
if not cands:
    print(f"ERROR: no {ckpt_prefix}_<step>.pth in {ckpt_folder}")
    sys.exit(1)
_, ckpt = max(cands)

for p in (trace, fa, fi, ckpt, cb_wkv, cb_shift):
    if not os.path.exists(p):
        print(f"ERROR: missing {p}")
        sys.exit(1)


def by_user_mean(path):
    tot, n = 0.0, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tot += r["metrics"]["LogLoss"]
        n += 1
    return tot / n, n


ahead, n_a = by_user_mean(fa)
imm, n_i = by_user_mean(fi)
print(f"[finish] {name}: ahead {ahead:.6f} (n={n_a})  imm {imm:.6f} (n={n_i})")
if n_a != expect_n or n_i != expect_n:
    print(f"ERROR: expected n={expect_n} users in both modes (sharded-eval merge incomplete?)")
    sys.exit(1)

r = subprocess.run([sys.executable, os.path.join(ROOT, "optimization", "promote_champion_5k.py"),
                    "--name", name, "--trace", trace,
                    "--final-ahead", f"{ahead:.6f}", "--final-imm", f"{imm:.6f}",
                    "--ckpt", ckpt, "--cb-wkv", cb_wkv, "--cb-shift", cb_shift])
sys.exit(r.returncode)
