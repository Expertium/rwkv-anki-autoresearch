"""How much does POOLING cost? Per-layer vs pooled shift bases, from an existing shift dump.

The artifact question is not academic: `fake_pq_shift` in `rwkv/model/rwkv_model.py` takes only a ROLE
(0 = t_xshift, 1 = c_xshift), i.e. the existing shift codebook is pooled over every stream and layer.
If a fixed basis can be pooled the same way, the whole shift codec needs TWO 32x(m+1) matrices and no
per-layer or per-stream lookup anywhere - no engine map, no source edit, no stream identity in the
Python forward. If it cannot, because the per-layer MEANS differ (the mean carries ~70% of the energy),
the artifact needs a key per (stream, role, layer) and the channel mixer's layer id has to be found.

This script decides it with the existing dump, by scoring the same vectors three ways at the same m:
  per-layer   each key's own mean + basis (the upper bound the artifact could ever reach);
  pooled-P    one basis pooled over the keys, but each key keeps its OWN mean;
  pooled-all  one mean AND one basis pooled over the keys (what a role-only artifact would do).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/analyze_shift_pooling.py \
      --npz scratchpad/quant_plan/corruption/shift_probe_out/shifts_note_l2i8s4.npz [--m 8]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rel_err(x, x_hat):
    return float(((x - x_hat) ** 2).sum(axis=1).mean() / max((x ** 2).sum(axis=1).mean(), 1e-30))


def quant_rows(v, qmax=127.0):
    amax = np.abs(v).max(axis=1, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    return np.clip(np.round(v / scale), -qmax, qmax) * scale


def top_modes(cent, m):
    cov = cent.T @ cent / max(cent.shape[0] - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    return evec[:, np.argsort(ev)[::-1]][:, :m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--m", default="6,8,10,12")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ms = [int(v) for v in args.m.split(",")]

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    keys, xs = [], []
    for i, key in enumerate(meta["keys"]):
        x = doc[f"a{i}"].astype(np.float32)
        keys.append(key)
        xs.append(x)
    n = min(x.shape[0] for x in xs)
    xs = [x[:n] for x in xs]
    print(f"dump {p.name}  keys={keys}  n per key={n}")

    mus = [x.mean(axis=0) for x in xs]
    print("\nper-key mean norms and pairwise cosines (the mean is ~70% of the energy):")
    for i, k in enumerate(keys):
        print(f"  {k:>24}  |mu|={np.linalg.norm(mus[i]):.3f}  "
              f"|mu|/rms={np.linalg.norm(mus[i]) / np.sqrt((xs[i] ** 2).sum(1).mean() / 32):.3f}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = float(mus[i] @ mus[j] / max(np.linalg.norm(mus[i]) * np.linalg.norm(mus[j]), 1e-30))
            print(f"  cos(mu[{keys[i]}], mu[{keys[j]}]) = {c:.4f}")

    pooled_all = np.concatenate(xs)
    mu_all = pooled_all.mean(axis=0)
    cent_all = (pooled_all - mu_all).astype(np.float64)
    print(f"\n{'m':>3} {'bits/vec':>8} | " + " ".join(f"{k:>22}" for k in keys)
          + f" | {'POOLED-P':>9} {'POOLED-ALL':>10}")
    out = {}
    for m in ms:
        row = {}
        p_all = top_modes(cent_all, m)
        cells = []
        for i, k in enumerate(keys):
            cent = (xs[i] - mus[i]).astype(np.float64)
            p_i = top_modes(cent, m)
            rec_i = mus[i] + quant_rows((xs[i] - mus[i]) @ p_i) @ p_i.T
            rec_pp = mus[i] + quant_rows((xs[i] - mus[i]) @ p_all) @ p_all.T
            rec_pa = mu_all + quant_rows((xs[i] - mu_all) @ p_all) @ p_all.T
            row[k] = {"per_layer": rel_err(xs[i], rec_i.astype(np.float32)),
                      "pooled_P": rel_err(xs[i], rec_pp.astype(np.float32)),
                      "pooled_all": rel_err(xs[i], rec_pa.astype(np.float32))}
            cells.append(f"{row[k]['per_layer']:.5f}/{row[k]['pooled_P']:.5f}/{row[k]['pooled_all']:.5f}")
        print(f"{m:>3} {m * 8:>8} | " + " ".join(f"{c:>22}" for c in cells))
        out[m] = row
    print("\n(cells are per_layer / pooled_P / pooled_all   - relative reconstruction error, int8 coefs)")
    print("DECISION: if pooled_P is within a few percent of per_layer, the artifact can be PER ROLE only "
          "(pooled over layers AND streams), which needs no stream identity in Python and no per-layer "
          "engine map. If pooled_P degrades, the mean must be per layer.")
    if args.out:
        Path(args.out).write_text(json.dumps({"npz": str(p), "keys": keys, "by_m": out}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
