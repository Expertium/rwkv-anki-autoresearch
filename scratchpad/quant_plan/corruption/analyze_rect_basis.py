"""Is the fixed-basis (codec) state block best as a SQUARE m x m, or as an m x m' rectangle?

The codec stores the m x m block C = P_m^T W Q_m, so its payload is m*m numbers. Every rung measured
so far uses the SAME m on both sides, which silently assumes the row side and the column side of the
state need the same number of modes. They need not: W is a decayed sum of k (x) v projections, so the
row side is built from one projected feature and the column side from another, and the two can have
very different spectra.

The question is only meaningful at an EQUAL payload, so this reads an existing raw-state dump and, for
every payload P, reports the best RECTANGLE with a*b = P beside its square counterpart - in-sample and
leave-one-user-out. A rectangle that beats the square at the same payload is a free win (same bits,
more captured energy), and it needs an engine/scope spelling for an asymmetric m but no new fit.

Reads the dump written by `state_probe.py` (the same file `fit_fixedbasis.py` consumes). CPU-only,
seconds, cannot touch the GPU.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/analyze_rect_basis.py \
      --npz scratchpad/quant_plan/corruption/state_probe_out/states_l2i8s4all.npz \
      [--payloads 16,20,25,30,36,42,49,64] [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

K = 32
MAX_MODES = 16


def fit(x, m):
    n = x.shape[0]
    crow = np.einsum("nij,nkj->ik", x, x) / n
    ccol = np.einsum("nij,nil->jl", x, x) / n
    _, prow = np.linalg.eigh(crow)
    _, pcol = np.linalg.eigh(ccol)
    return prow[:, ::-1][:, :m].astype(np.float32), pcol[:, ::-1][:, :m].astype(np.float32)


def capture(x, p, q):
    tot = float((x * x).sum(axis=(1, 2)).mean())
    c = np.einsum("ik,nij,jl->nkl", p, x, q)
    return float((c * c).sum(axis=(1, 2)).mean() / max(tot, 1e-30))


def factor_pairs(payload, cap=MAX_MODES):
    """Every (a, b) with a*b == payload and a <= b <= cap."""
    out = []
    for a in range(1, cap + 1):
        if payload % a == 0:
            b = payload // a
            if a <= b <= cap:
                out.append((a, b))
    return out


def loo_capture(x, u, a, b, m=MAX_MODES):
    users = sorted({int(v) for v in u})
    if len(users) < 2:
        return None
    vals = []
    for held in users:
        tr, te = x[u != held], x[u == held]
        if tr.shape[0] < 8 or te.shape[0] < 1:
            continue
        p, q = fit(tr, m)
        vals.append(capture(te, p[:, :a], q[:, :b]))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--streams", default="1:note,2:card",
                    help="rank:stream, used when the dump key has no stream name")
    ap.add_argument("--payloads", default="16,20,25,30,36,42,49,64")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    by_rank = {}
    for pair in args.streams.split(","):
        r, _, s = pair.strip().partition(":")
        by_rank[int(r)] = s.strip()
    payloads = [int(v) for v in args.payloads.split(",")]

    groups, users = {}, {}
    for i, key in enumerate(meta["keys"]):
        parts = key.split("|")
        stream = parts[0]
        if stream == "x":
            stream = by_rank.get(int(parts[1][1:]), parts[1])
        lay = int(parts[2][1:])
        x = doc[f"a{i}"].astype(np.float32)
        un = f"u{i}"
        u = doc[un] if un in doc.files else np.full(x.shape[0], -1, dtype=np.int32)
        groups.setdefault((stream, lay), []).append(x)
        users.setdefault((stream, lay), []).append(u)

    report = {"npz": str(p), "payloads": payloads, "max_modes": MAX_MODES, "by_key": {}}
    for (stream, lay), chunks in sorted(groups.items()):
        x = np.concatenate(chunks)
        u = np.concatenate(users[(stream, lay)])
        pv, qv = fit(x, MAX_MODES)
        rows = []
        print(f"\n=== {stream} L{lay}  N={x.shape[0]} ===")
        print(f"{'payload':>7} {'m x m2':>7} {'in-sample':>9} {'LOO':>9}   note")
        for pay in payloads:
            cands = factor_pairs(pay)
            if not cands:
                continue
            scored = []
            for a, b in cands:
                ins = capture(x, pv[:, :a], qv[:, :b])
                oo = loo_capture(x, u, a, b)
                scored.append({"a": a, "b": b, "in_sample": ins, "loo": oo})
            best = max(scored, key=lambda r: (r["loo"] if r["loo"] is not None else r["in_sample"]))
            sq = [r for r in scored if r["a"] == r["b"]]
            sqtxt = (f"square m={sq[0]['a']} in {sq[0]['in_sample']:.4f} "
                     f"loo {sq[0]['loo']:.4f}" if sq else "no square at this payload")
            all_txt = ", ".join(
                "%dx%d:%.4f/%s" % (r["a"], r["b"], r["in_sample"],
                                   "n/a" if r["loo"] is None else "%.4f" % r["loo"])
                for r in scored)
            print("%7d %7s %9.4f %9.4f   ALL: %s  | %s"
                  % (pay, "%dx%d" % (best["a"], best["b"]), best["in_sample"],
                     float("nan") if best["loo"] is None else best["loo"], all_txt, sqtxt))
            rows.append({"payload": pay, "candidates": scored, "best": best,
                         "square": sq[0] if sq else None})
        report["by_key"][f"{stream}__L{lay}"] = {"n": int(x.shape[0]), "rows": rows}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
