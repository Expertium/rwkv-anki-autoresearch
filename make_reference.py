"""Build the Rust-parity reference from the clean 100-user model.

Runs Python RNN-mode inference (run_as_rnn) on small held-out users with a FIXED per-user
seed (so the otherwise-random id-encodings are reproducible) and saves per-user + by-user
mean ahead/imm LogLoss. The Rust port must match the saved mean within +/-0.0005.

The 5 smallest users in 101-200 are used (RNN path is ~20 rev/s; giants like 112=790k are
infeasible). Expand REF_USERS later for a stricter gate.
"""
import ast
import glob
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

torch.set_num_threads(7)

import rwkv.run_as_rnn as rnn

DATA = Path("../anki-revlogs-10k")
LABEL_DB = "label_filter_db"
LABEL_DB_SIZE = 2_000_000_000
REF_USERS = [107, 136, 156]  # 3 smallest in 101-200 (fast; expand for a stricter gate)
OUT_DIR = "pretrain/rwkv/ref_100"


def latest_checkpoint():
    cks = [
        p for p in glob.glob(f"{OUT_DIR}/rwkv_ref_*.pth") if "optim" not in p
    ]
    # pick the highest step
    return max(cks, key=lambda p: int(re.search(r"_(\d+)\.pth$", p).group(1)))


def parse_stats(text):
    lines = text.splitlines()
    imm = ahead = None
    for i, ln in enumerate(lines):
        if ln.strip() == "RWKV-P:" and i + 1 < len(lines):
            imm = ast.literal_eval(lines[i + 1])
        if ln.strip() == "RWKV:" and i + 1 < len(lines):
            ahead = ast.literal_eval(lines[i + 1])
    return imm, ahead


def main():
    model_path = latest_checkpoint()
    print(f"reference checkpoint: {model_path}", flush=True)
    rows = []
    for u in REF_USERS:
        torch.manual_seed(u)  # reproducible id-encodings for this user
        buf = io.StringIO()
        with redirect_stdout(buf):
            rnn.run(DATA, model_path, LABEL_DB, LABEL_DB_SIZE, u, verbose=False)
        imm, ahead = parse_stats(buf.getvalue())
        rows.append({"user": u, "size": imm["size"],
                     "imm_logloss": imm["metrics"]["LogLoss"],
                     "ahead_logloss": ahead["metrics"]["LogLoss"]})
        print(f"user {u}: size {imm['size']}, imm {imm['metrics']['LogLoss']:.5f}, "
              f"ahead {ahead['metrics']['LogLoss']:.5f}", flush=True)

    mean_imm = sum(r["imm_logloss"] for r in rows) / len(rows)
    mean_ahead = sum(r["ahead_logloss"] for r in rows) / len(rows)
    ref = {
        "model": os.path.basename(model_path),
        "seed_scheme": "torch.manual_seed(user_id) per user",
        "users": REF_USERS,
        "per_user": rows,
        "mean_imm_logloss": mean_imm,
        "mean_ahead_logloss": mean_ahead,
        "tolerance": 0.0005,
        "note": "Rust RNN port must match mean_*_logloss within +/-tolerance.",
    }
    out = f"{OUT_DIR}/ref_metrics.json"
    with open(out, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"\nMEAN imm={mean_imm:.5f}  ahead={mean_ahead:.5f}", flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
