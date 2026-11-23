"""Can the two token-shift vectors of a layer be coded as ONE vector plus a small correction?

Every stream layer persists TWO 32-d shift vectors (`LayerState` = t_xshift + t_state + c_xshift),
and they are not independent: both are `layer_norm(x)` of the SAME layer input, gathered at the
previous timestep - one by the WKV mixer, one by the channel mixer. If their two bases and their
learned gains are similar, the pair carries far less than 2*32 independent numbers, and the right
code is not "two separate m-mode projections" but "one m_t-mode projection of t, plus m_r modes of the
residual (c - t)". That matters because a residual basis can capture a near-copy for almost nothing,
while a separate projection of c must respend m modes on directions it shares with t.

This reads the dump written by `corruption/shift_probe.py` and answers it with the SAME measure the
rest of this line uses - the relative reconstruction error under a per-vector amax int8 round trip -
so the numbers are directly comparable to the "int4 now" column (2 vectors x 32 coords x 4 bits = 256
bits/layer). It also reports the plain correlation, which is the one number that decides whether the
pair idea is alive at all.

CPU-only, seconds, cannot touch the GPU.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/analyze_shift_pairs.py \
      --npz scratchpad/quant_plan/corruption/shift_probe_out/shifts_note_l2i8s4.npz [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

M_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 16)
RESID_GRID = (0, 1, 2, 3, 4, 6)


def rel_err(x, x_hat):
    num = ((x - x_hat) ** 2).sum(axis=1).mean()
    den = (x ** 2).sum(axis=1).mean()
    return float(num / max(den, 1e-30))


def quant_rows(v, qmax):
    amax = np.abs(v).max(axis=1, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    return np.clip(np.round(v / scale), -qmax, qmax) * scale


def fit_basis(x, m):
    """Mean + top-m eigenvectors of the covariance (the model-side pair ships both)."""
    mu = x.mean(axis=0)
    cent = (x - mu).astype(np.float64)
    cov = cent.T @ cent / max(x.shape[0] - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    return mu.astype(np.float32), evec[:, np.argsort(ev)[::-1]][:, :m].astype(np.float32)


def project_err(x, mu, p, qmax=127.0):
    rec = mu + quant_rows((x - mu) @ p, qmax) @ p.T
    return rel_err(x, rec.astype(np.float32))


def loo(x, u, m, qmax=127.0):
    users = sorted({int(v) for v in u})
    if len(users) < 2:
        return None
    out = []
    for held in users:
        tr, te = x[u != held], x[u == held]
        if tr.shape[0] < 8 or te.shape[0] < 1:
            continue
        mu, p = fit_basis(tr, m)
        out.append(project_err(te, mu, p, qmax))
    return float(np.mean(out)) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    groups = {}
    for i, key in enumerate(meta["keys"]):
        info = meta["meta"][key]
        role = "t" if "Time" in info["role"] else ("c" if "Channel" in info["role"] else info["role"])
        un = f"u{i}"
        u = doc[un] if un in doc.files else np.full(doc[f"a{i}"].shape[0], -1, dtype=np.int32)
        groups.setdefault((info["stream"], info["layer"]), {})[role] = (
            doc[f"a{i}"].astype(np.float32), u, info["role"])

    report = {"npz": str(p), "tag": meta.get("tag"), "stream": meta.get("stream"),
              "bits_now_per_layer": 2 * 32 * 4, "by_pair": {}}
    for (stream, lay), roles in sorted(groups.items()):
        if "t" not in roles or "c" not in roles:
            print(f"\n=== {stream} L{lay}: only roles {sorted(roles)} present - nothing to pair ===")
            continue
        t, ut, role_t = roles["t"]
        c, uc, role_c = roles["c"]
        n = min(t.shape[0], c.shape[0])
        t, c, ut = t[:n], c[:n], ut[:n]
        cos = float(np.mean((t * c).sum(axis=1) /
                           np.maximum(np.linalg.norm(t, axis=1) * np.linalg.norm(c, axis=1), 1e-30)))
        resid_ratio = rel_err(c, t)
        d = (c - t).astype(np.float32)
        rec = {"n": int(n), "role_t": role_t, "role_c": role_c, "cosine": cos,
               "resid_energy_ratio": resid_ratio, "separate": {}, "pair": {}, "int4_now": {}}
        print(f"\n=== {stream} L{lay}  roles {role_t} vs {role_c}  N={n} ===")
        print(f"    cosine(t, c) = {cos:.4f}   ||c - t||^2 / ||c||^2 = {resid_ratio:.4f}")
        print(f"{'bits/layer':>10} {'scheme':>34} {'err_t':>8} {'err_c':>8} {'worst':>8}")
        print("-" * 72)
        e_t_int4, e_c_int4 = rel_err(t, quant_rows(t, 7.0)), rel_err(c, quant_rows(c, 7.0))
        print(f"{2 * 32 * 4:>10} {'int4 now: 32 coords x 4 bits each':>34} {e_t_int4:>8.5f} "
              f"{e_c_int4:>8.5f} {max(e_t_int4, e_c_int4):>8.5f}")
        rec["int4_now"] = {"bits": 2 * 32 * 4, "err_t": e_t_int4, "err_c": e_c_int4}

        mut, pt = fit_basis(t, max(M_GRID))
        for m in M_GRID:
            e_t = project_err(t, mut, pt[:, :m])
            e_c = project_err(c, mut, pt[:, :m])          # c coded against t's own basis
            bits = 2 * m * 8
            rec["separate"][m] = {"bits": bits, "err_t": e_t, "err_c_using_t_basis": e_c,
                                  "loo_t": loo(t, ut, m), "loo_c": loo(c, uc, m)}
            print(f"{bits:>10} {'separate basis, {} modes each'.format(m):>34} {e_t:>8.5f} "
                  f"{e_c:>8.5f} {max(e_t, e_c):>8.5f}")

        for mt in (4, 6, 8):
            for mr in RESID_GRID:
                mu_t, p_t = fit_basis(t, mt)
                t_hat = (mu_t + quant_rows((t - mu_t) @ p_t) @ p_t.T).astype(np.float32)
                if mr == 0:
                    c_hat = t_hat
                else:
                    mu_d, p_d = fit_basis(d, mr)
                    c_hat = (t_hat + mu_d + quant_rows((d - mu_d) @ p_d) @ p_d.T).astype(np.float32)
                e_t = rel_err(t, t_hat)
                e_c = rel_err(c, c_hat)
                bits = (mt + mr) * 8
                rec["pair"][f"{mt}+{mr}"] = {"bits": bits, "err_t": e_t, "err_c": e_c,
                                             "worst": max(e_t, e_c)}
                print(f"{bits:>10} {f't: {mt} modes + (c - t): {mr} modes':>34} {e_t:>8.5f} "
                      f"{e_c:>8.5f} {max(e_t, e_c):>8.5f}")

        best = min(rec["pair"].items(), key=lambda kv: (kv[1]["worst"], kv[1]["bits"]))
        target = max(e_t_int4, e_c_int4)
        winners = {k: v for k, v in rec["pair"].items()
                   if v["worst"] <= target and v["bits"] < 2 * 32 * 4}
        rec["pair_beats_int4"] = winners
        print(f"    -> pair codes that beat int4's error ({target:.5f}) at FEWER than 256 bits: "
              f"{sorted((k, v['bits'], round(v['worst'], 5)) for k, v in winners.items()) or 'NONE'}"
              f"; best pair = {best[0]} at {best[1]['bits']} bits, worst {best[1]['worst']:.5f}")
        report["by_pair"][f"{stream}__L{lay}"] = rec

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
