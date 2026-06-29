"""Viability screen for low-rank card WKV state (step 4 / the 0.15 KB target).

Shells out to the rust engine's --dump-card-state for a sample of (user, card_pos), parses the 32x32
fp32 WKV matrix, and computes its singular values. Reports how much spectral energy sits in the top-1
/ top-2 / top-4 singular values -> tells us whether rank-1/rank-2 factoring of the card state is viable
(high top-1 energy = rank-1 stores most of the state) BEFORE building the recurrence/QAT machinery.

energy_r = sum(sigma_i^2 for i<r) / sum(sigma_i^2)  (fraction of Frobenius energy in rank r).
A rank-r approx's relative Frobenius error = sqrt(1 - energy_r).
Run from repo root. Uses the deployed champion weights by default.
"""
import os
import re
import subprocess
import sys

import numpy as np

BIN = r"rust\rwkv-infer\target\release\rwkv-infer.exe"
W = sys.argv[1] if len(sys.argv) > 1 else "reference/rwkv_iter45.safetensors"
USERS = [107, 121, 136, 156]
CARD_POS = [0, 5, 20, 60, 150]


def dump_matrix(user, pos):
    env = dict(os.environ, RWKV_WEIGHTS=W)
    out = subprocess.run([BIN, "--dump-card-state", str(user), str(pos)],
                         capture_output=True, text=True, env=env).stdout
    lines = out.splitlines()
    # find the fp32 matrix block, parse the next 32 rows of 32 floats
    cid = None
    for i, ln in enumerate(lines):
        m = re.search(r"dense card id (\d+)", ln)
        if m:
            cid = int(m.group(1))
        if ln.startswith("=== fp32") and "WKV state" in ln:
            rows = []
            for r in range(i + 1, i + 1 + 40):
                if r >= len(lines):
                    break
                nums = re.findall(r"-?\d+\.\d+", lines[r])
                if len(nums) >= 8:
                    rows.append([float(x) for x in nums])
                elif rows:
                    break
            mat = np.array(rows, dtype=np.float64)
            return cid, mat
    return cid, None


def main():
    seen = set()
    e1, e2, e4 = [], [], []
    n_mats = 0
    for u in USERS:
        for p in CARD_POS:
            cid, mat = dump_matrix(u, p)
            if mat is None or mat.shape[0] < 8 or mat.shape[0] != mat.shape[1]:
                continue
            key = (u, cid)
            if key in seen:
                continue
            seen.add(key)
            s = np.linalg.svd(mat, compute_uv=False)
            energy = s ** 2
            tot = energy.sum()
            if tot <= 0:
                continue
            cum = np.cumsum(energy) / tot
            e1.append(cum[0]); e2.append(cum[1]); e4.append(cum[3])
            n_mats += 1
            if n_mats <= 12:
                print(f"u{u} card{cid}: shape {mat.shape}  top sigma {s[:4].round(3)}  "
                      f"energy r1={cum[0]:.3f} r2={cum[1]:.3f} r4={cum[3]:.3f}")
    if not n_mats:
        print("no matrices parsed"); return
    e1, e2, e4 = np.array(e1), np.array(e2), np.array(e4)
    print(f"\n=== {n_mats} distinct card states ===")
    for name, arr in [("rank-1", e1), ("rank-2", e2), ("rank-4", e4)]:
        relerr = np.sqrt(np.clip(1 - arr, 0, 1))
        print(f"{name}: energy mean {arr.mean():.4f} (min {arr.min():.4f})  "
              f"-> Frobenius relerr mean {relerr.mean():.4f} (max {relerr.max():.4f})")


if __name__ == "__main__":
    main()
