"""Is a plain MEAN SUBTRACTION enough? The cheapest possible shift coder, priced against the basis.

The shift vector is `layer_norm(previous input)`, and its MEAN carries 68-71% of the energy. The deploy
coder (`quant_vec_inplace`) scales by the vector's own amax, so most of the amax is spent on the part
that never changes: subtracting a fixed, model-side mean first (free in state bits - it ships with the
model) leaves a much smaller residual for the same int-N grid, which should cut the error by roughly
the residual's energy share. That would be a far smaller change than a basis coder: no projection, no
m coefficients, no matrix in the artifact - just `x -> x - mu` before the existing round trip.

This scores, per key, at each int-N width: the raw vector (what the engine does today) and the
mean-subtracted residual, both by the same per-vector amax rule, plus the same numbers for the
PER-ROLE pooled mean (the artifact a role-only design would ship).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/analyze_shift_meansub.py \
      --npz scratchpad/quant_plan/corruption/shift_probe_out/shifts_note_l2i8s4.npz [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

QMAX = {8: 127.0, 6: 31.0, 5: 15.0, 4: 7.0, 3: 3.0, 2: 1.0}


def rel_err(x, x_hat):
    return float(((x - x_hat) ** 2).sum(axis=1).mean() / max((x ** 2).sum(axis=1).mean(), 1e-30))


def quant_rows(v, qmax):
    amax = np.abs(v).max(axis=1, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    return np.clip(np.round(v / scale), -qmax, qmax) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    xs = [doc[f"a{i}"].astype(np.float32) for i in range(len(meta["keys"]))]
    keys = meta["keys"]
    n = min(x.shape[0] for x in xs)
    xs = [x[:n] for x in xs]
    mu_role = np.concatenate(xs).mean(axis=0)          # one mean for the whole role (pooled)
    print(f"dump {p.name}  keys={keys}  n/key={n}  |mu_role|={np.linalg.norm(mu_role):.3f}")

    report = {"npz": str(p), "keys": keys, "by_key": {}}
    for i, k in enumerate(keys):
        x = xs[i]
        mu_key = x.mean(axis=0)
        d_key = x - mu_key
        d_role = x - mu_role
        print(f"\n=== {k} ===")
        print(f"{'scheme':>34} {'bits/vec':>8} {'rel_err':>9}")
        rec = {}
        for q in (4, 3, 2):
            e_raw = rel_err(x, quant_rows(x, QMAX[q]).astype(np.float32))
            e_key = rel_err(x, (mu_key + quant_rows(d_key, QMAX[q])).astype(np.float32))
            e_role = rel_err(x, (mu_role + quant_rows(d_role, QMAX[q])).astype(np.float32))
            rec[f"int{q}"] = {"bits": 32 * q, "raw": e_raw, "mu_per_key": e_key,
                              "mu_per_role": e_role}
            print(f"{'int' + str(q) + ' RAW (what ships today)':>34} {32 * q:>8} {e_raw:>9.5f}")
            print(f"{'int' + str(q) + ' on (x - mu_key)':>34} {32 * q:>8} {e_key:>9.5f}")
            print(f"{'int' + str(q) + ' on (x - mu_role)':>34} {32 * q:>8} {e_role:>9.5f}")
        report["by_key"][k] = rec
    print("\nReading: the mean is FREE in state bits (it ships with the model), so the 'intN on "
          "(x - mu)' rows cost exactly what the 'intN RAW' rows cost. A mean-subtracted int3 that beats "
          "raw int4 saves 32 bits/vector with a lower error, and needs no basis at all.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
