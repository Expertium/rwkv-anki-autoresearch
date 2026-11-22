"""Analyze a raw-state dump: how many bits does a FIXED-basis state scheme need?

Reads the .npz written by `state_probe.py` (raw 32x32 WKV states per stream/layer, with the user id of
each state) and answers the question the current rank-r scheme never asks: are the state's dominant
directions SHARED across users and time, or does every step need a fresh basis?

Representations scored, all on the same states, all by the same measure - the mean fraction of
||W||_F^2 the representation preserves:

  SVD rank-r          the CURRENT scheme: U_r V_r^T, optimal per state. 2*r*K numbers per layer.
  2D-PCA (m)          a FIXED row basis P (top-m eigenvectors of E[W W^T]) and a FIXED column basis
                      Q (top-m of E[W^T W]); the stored object is P^T W Q, m*m numbers.
  vector PCA (m)      the state as a 1024-vector projected onto the top-m eigenvectors of its own
                      covariance; the stored object is m numbers.
  mean only           E[W] alone, 0 numbers (the mean would live in the model weights).

THE IN-SAMPLE CURVE IS NOT THE ANSWER. A basis fitted on the same states it is scored on is
optimistic; what matters is whether it transfers to a user it was never fitted on. So when the dump
carries user ids, every fixed-basis number is ALSO reported leave-one-user-out: fit on all other
users, score on the held-out one. A scheme whose held-out capture collapses is not a scheme.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/analyze_state_subspace.py \
      --npz scratchpad/quant_plan/corruption/state_probe_out/states_<tag>.npz \
      [--out scratchpad/quant_plan/corruption/state_subspace_<tag>.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

K = 32
SHIFT_BITS = 4          # the shift vectors stay at int4 in every scheme compared here
QBITS = 8               # factor/coefficient precision for the bit table
CURVE_R = tuple(range(1, 9))
CURVE_M = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
LOO_M = (2, 4, 8, 16, 32)


def svd_curve(x):
    """Mean captured energy fraction of the optimal rank-r truncation, r = 1..8."""
    s = np.linalg.svd(x, compute_uv=False)
    e = s * s
    tot = e.sum(axis=-1, keepdims=True)
    cum = np.cumsum(e, axis=-1) / np.maximum(tot, 1e-30)
    return {r: float(cum[:, r - 1].mean()) for r in CURVE_R}


def fit_tdpca(x, m):
    n = x.shape[0]
    crow = np.einsum("nij,nkj->ik", x, x) / n          # E[W W^T]
    ccol = np.einsum("nij,nil->jl", x, x) / n          # E[W^T W]
    _, prow = np.linalg.eigh(crow)
    _, pcol = np.linalg.eigh(ccol)
    return prow[:, ::-1][:, :m], pcol[:, ::-1][:, :m]  # descending


def capture_tdpca(x, p, q):
    tot = float((x * x).sum(axis=(1, 2)).mean())
    cap = np.einsum("ik,nij,jl->nkl", p, x, q)
    return float((cap * cap).sum(axis=(1, 2)).mean() / max(tot, 1e-30))


def fit_vpca(x, m):
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float64)
    mean = flat.mean(axis=0)
    cent = flat - mean
    cov = cent.T @ cent / max(n - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    return mean, evec[:, order][:, :m], float(ev[order].sum())


def capture_vpca(x, mean, basis):
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float64)
    tot = float((flat * flat).sum(axis=1).mean())
    c = (flat - mean) @ basis
    return float((c * c).sum(axis=1).mean() / max(tot, 1e-30))


def tdpca_curve(x):
    tot = float((x * x).sum(axis=(1, 2)).mean())
    out = {}
    for m in CURVE_M:
        if m > K:
            break
        p, q = fit_tdpca(x, m)
        out[m] = capture_tdpca(x, p, q)
    return out


def vpca_curve(x):
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float64)
    mean = flat.mean(axis=0)
    cent = flat - mean
    cov = cent.T @ cent / max(n - 1, 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    tot = float(ev.sum())
    cum = np.cumsum(ev) / max(tot, 1e-30)
    out = {m: float(cum[m - 1]) for m in CURVE_M}
    energy = float((flat * flat).sum(axis=1).mean())
    mean_share = float((mean * mean).sum() / max(energy, 1e-30))
    return out, mean_share, tot


def loo_curves(x, u, ms=LOO_M):
    """Leave-one-user-out capture for both fixed-basis schemes (fit on the rest, score on the one)."""
    users = sorted({int(v) for v in u})
    if len(users) < 2:
        return None
    td, vp = {}, {}
    for m in ms:
        tds, vps = [], []
        for held in users:
            tr = x[u != held]
            te = x[u == held]
            if tr.shape[0] < 8 or te.shape[0] < 1:
                continue
            p, q = fit_tdpca(tr, m)
            tds.append(capture_tdpca(te, p, q))
            mean, basis, _ = fit_vpca(tr, m)
            vps.append(capture_vpca(te, mean, basis))
        td[m] = {"mean": float(np.mean(tds)), "min": float(np.min(tds)),
                 "per_user": [float(v) for v in tds]}
        vp[m] = {"mean": float(np.mean(vps)), "min": float(np.min(vps)),
                 "per_user": [float(v) for v in vps]}
    return {"tdpca": td, "vpca": vp, "users": users}


def bits_svd(r):
    return 2 * r * K * QBITS + 64 * SHIFT_BITS


def bits_fixed(m):
    return m * QBITS + 64 * SHIFT_BITS


def bits_tdpca(m):
    return m * m * QBITS + 64 * SHIFT_BITS


def need(curve, bitsf, target):
    for m in sorted(curve):
        if curve[m] >= target:
            return {"m": m, "captured": curve[m], "bits": bitsf(m)}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--target", type=float, default=0.9938,
                    help="captured-energy target to report the required bits for "
                         "(default 0.9938 = the card's measured rank-2 SVD value)")
    args = ap.parse_args()

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    report = {"npz": str(p), "tag": meta.get("tag"), "every": meta.get("every"),
              "target": args.target, "qbits": QBITS, "shift_bits": SHIFT_BITS, "by_key": {}}

    print(f"dump {p}  tag={meta.get('tag')}  every={meta.get('every')}  "
          f"total states={meta.get('total_states')}")
    head = (f"{'key':>12} {'N':>6} {'users':>5} {'mean':>6} | {'svd r1':>7} {'r2':>7} | "
            f"{'2dpca m4':>8} {'m8':>7} {'m16':>7} | {'vpca m4':>7} {'m8':>7} {'m16':>7} | "
            f"{'LOO 2d m8':>9} {'m16':>7} | {'LOO vp m8':>9} {'m16':>7}")
    print(head)
    print("-" * len(head))

    for i, key in enumerate(meta["keys"]):
        x = doc[f"a{i}"].astype(np.float32)
        uname = f"u{i}"
        u = doc[uname] if uname in doc.files else np.full(x.shape[0], -1, dtype=np.int32)
        n = x.shape[0]
        energy = float((x * x).sum(axis=(1, 2)).mean())
        sv = svd_curve(x)
        td = tdpca_curve(x)
        vp, mean_share, vtot = vpca_curve(x)
        loo = loo_curves(x, u)

        rec = {"n": n, "energy": energy, "svd": sv, "tdpca": td, "vpca": vp,
               "vpca_mean_share": mean_share, "vpca_total_var": vtot, "loo": loo,
               "users": meta.get("users", {}).get(key),
               "bits_svd": {str(r): bits_svd(r) for r in CURVE_R},
               "bits_tdpca": {str(m): bits_tdpca(m) for m in td},
               "bits_vpca": {str(m): bits_fixed(m) for m in vp},
               "rank_now": int(meta["meta"][key]["rank"]),
               "bits_now": bits_svd(int(meta["meta"][key]["rank"])),
               "need_svd": need(sv, bits_svd, args.target),
               "need_tdpca": need(td, bits_tdpca, args.target),
               "need_vpca": need(vp, bits_fixed, args.target)}
        if loo:
            rec["need_tdpca_loo"] = need({m: v["mean"] for m, v in loo["tdpca"].items()},
                                         bits_tdpca, args.target)
            rec["need_vpca_loo"] = need({m: v["mean"] for m, v in loo["vpca"].items()},
                                        bits_fixed, args.target)
        report["by_key"][key] = rec

        l8 = loo["tdpca"][8]["mean"] if loo else float("nan")
        l16 = loo["tdpca"][16]["mean"] if loo else float("nan")
        w8 = loo["vpca"][8]["mean"] if loo else float("nan")
        w16 = loo["vpca"][16]["mean"] if loo else float("nan")
        print(f"{key:>12} {n:>6} {len(meta.get('users', {}).get(key, [])):>5} "
              f"{mean_share:>6.3f} | {sv[1]:>7.4f} {sv[2]:>7.4f} | {td.get(4, float('nan')):>8.4f} "
              f"{td.get(8, float('nan')):>7.4f} {td.get(16, float('nan')):>7.4f} | "
              f"{vp.get(4, float('nan')):>7.4f} {vp.get(8, float('nan')):>7.4f} "
              f"{vp.get(16, float('nan')):>7.4f} | {l8:>9.4f} {l16:>7.4f} | {w8:>9.4f} {w16:>7.4f}")

    print(f"\nbits needed to capture >= {args.target:.4f} of E||W||^2 "
          f"(shift vectors 64*{SHIFT_BITS} = {64 * SHIFT_BITS} bits always included):")
    for key, rec in report["by_key"].items():
        r = rec["rank_now"]
        print(f"  {key:>12}  now svd r{r}: {rec['bits_now']:>5} bits ({rec['svd'][r]:.4f})"
              f"  | svd {rec['need_svd']} | 2dpca {rec['need_tdpca']} | vpca {rec['need_vpca']}")
        if "need_tdpca_loo" in rec:
            print(f"  {'':>12}  leave-one-user-out: 2dpca {rec['need_tdpca_loo']} | "
                  f"vpca {rec['need_vpca_loo']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
