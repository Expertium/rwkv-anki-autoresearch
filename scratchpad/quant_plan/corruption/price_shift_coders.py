"""Price the token-shift coders that are actually deployable, on the multi-user shift dump.

WHY THIS IS NOT `fit_shift_basis.py`. That script prices one artifact per key tier and prints a wide
grid, which is right for FITTING but noisy for DECIDING. This one answers one question with the numbers
that matter: at a given payload (bits per 32-d shift vector), which coder has the lowest relative
reconstruction error, and does the winner beat the deploy baseline (per-vector amax int4)?

THE CANDIDATES (all deployable: the model-side parts cost ZERO state bits).
  raw intN        the deploy baseline: one amax scale for the whole 32-vector, int-N.
  mean intN       subtract a fitted model-side mean, then raw intN on the residual.
  mX intN         `mu + P c`, P = top-m eigenvectors of the residual covariance, ONE amax scale for
                  all m coefficients (what `fixedbasis_vec_inplace` does today).
  mX intN +scale  the same, but every coefficient has its OWN model-side scale (its fitted std), so
                  the small modes are not squashed by the largest one - the PCA spectrum is wide, so
                  one shared scale throws away most of the resolution. Still ZERO state bits.

The scale variant is the one to watch: `mean int4` (128 bits) already beats `m16 int8` (128 bits), which
says the spectrum, not the mean, is what a single scale wastes.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/price_shift_coders.py \
      --npz scratchpad/quant_plan/corruption/shift_probe_out/shifts_note_multi6.npz \
            scratchpad/quant_plan/corruption/shift_probe_out/shifts_card_multi6.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

LEVEL = {"int8": (127.0, 8), "int6": (31.0, 6), "int4": (7.0, 4), "int3": (3.0, 3), "int2": (1.0, 2)}
MODES = (4, 6, 8, 12, 16, 24, 32)


def role_of(name):
    return 0 if "TimeMixer" in name else 1


def rel_err(x, x_hat):
    return float(((x - x_hat) ** 2).sum(1).mean() / max((x ** 2).sum(1).mean(), 1e-30))


def quant_rows(v, qmax):
    amax = np.abs(v).max(axis=1, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    return np.clip(np.round(v / scale), -qmax, qmax) * scale


def basis(x, m):
    mu = x.mean(0).astype(np.float32)
    c = (x - mu).astype(np.float64)
    ev, evec = np.linalg.eigh(c.T @ c / max(x.shape[0] - 1, 1))
    return mu, evec[:, np.argsort(ev)[::-1]][:, :m].astype(np.float32)


def load(npz_path):
    p = Path(npz_path)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    out = []
    for i, key in enumerate(meta["keys"]):
        info = meta["meta"][key]
        stream = info.get("stream") or key.split("|")[0]
        lay = int(info.get("layer", -1))
        out.append((key, stream, role_of(info.get("role", key.split("|")[1])), lay,
                    doc[f"a{i}"].astype(np.float32)))
    return out


def price(x, mu, p, qmax, bits, per_mode_scale):
    c = (x - mu) @ p
    if per_mode_scale:
        # PER-MODE AMPLITUDE: every coefficient gets its own model-side scale, so the small modes are
        # not squashed by the largest one. The scale must COVER the mode's range - a 1-sigma scale
        # clips ~32% of a Gaussian to +/- sigma, which is catastrophic (measured: 0.06-0.11, ~3x worse
        # than a single shared amax). `amax` never clips but wastes a little on the mode's own tail.
        s = np.abs(c).max(0) if per_mode_scale == "amax" else 3.0 * c.std(0)
        s = np.maximum(s.astype(np.float64), 1e-12)
        step = s / qmax
        cq = np.clip(np.round(c / step), -qmax, qmax) * step
    else:
        cq = quant_rows(c, qmax)
    return bits, rel_err(x, (mu + cq @ p.T).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=14, help="rows printed per payload (sorted by error)")
    args = ap.parse_args()
    groups = []
    for path in args.npz:
        groups.extend(load(path))

    for key, stream, role, lay, x in groups:
        print(f"\n=== {key}  stream={stream} role={role} layer={lay} rows={x.shape[0]} ===")
        rows = []  # (bits, err, label)
        for lvl, (q, b) in LEVEL.items():
            rows.append((32 * b, rel_err(x, quant_rows(x, q).astype(np.float32)), f"raw {lvl}"))
            mu0 = x.mean(0).astype(np.float32)
            rows.append((32 * b, rel_err(x, (mu0 + quant_rows(x - mu0, q)).astype(np.float32)),
                         f"mean {lvl}"))
        for m in MODES:
            mu, p = basis(x, m)
            for lvl, (q, b) in LEVEL.items():
                bits, err = price(x, mu, p, q, m * b, False)
                rows.append((bits, err, f"m{m} {lvl}"))
                bits, err = price(x, mu, p, q, m * b, "amax")
                rows.append((bits, err, f"m{m} {lvl} +mode-amax"))
                bits, err = price(x, mu, p, q, m * b, "std3")
                rows.append((bits, err, f"m{m} {lvl} +3sd"))
        seen = set()
        by_bits = {}
        for bits, err, label in sorted(rows):
            by_bits.setdefault(bits, []).append((err, label))
        for bits in sorted(by_bits):
            if bits in seen:
                continue
            seen.add(bits)
            top = sorted(by_bits[bits])[: args.top]
            print(f"  {bits:>4} bits/vector: " + "  ".join(f"{lbl}={e:.5f}" for e, lbl in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
