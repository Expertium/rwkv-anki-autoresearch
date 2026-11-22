"""Fit the FIXED bases for fixed-basis state coding from a raw-state dump, and report how they
generalize.

Reads the .npz written by `state_probe.py` (keys `<stream>|r<rank>|L<layer>` or `x|r<rank>|L<layer>`
when the probe could not name the stream, plus a per-state user id) and writes

    fixedbasis_<tag>.npz     P__<stream>__L<layer> / Q__<stream>__L<layer>   (K x M, M = --modes)
                             P__<stream>__ALL      / Q__<stream>__ALL        (pooled over the layers)

which is exactly what `rwkv/model/rwkv_ops.py`'s `_fb_lookup` reads when RWKV_FIXEDBASIS_PATH points
at it. The basis is fitted at --modes (default 16) so the SAME file serves any m <= that: the source
slices `[:, :m]`, and each candidate's QAT scope carries its own m.

HOW THE BASIS IS FITTED. P = the top-M eigenvectors of E[W W^T] (the row side) and Q = the top-M
eigenvectors of E[W^T W] (the column side), the standard separable 2D-PCA solution: those maximize
the expected captured energy ||P^T W Q||_F^2 over the state distribution.

WHAT IT REPORTS. For every key: the in-sample captured energy (the fit set is the score set - the
optimistic bound) AND the leave-one-user-out captured energy (fit on the other users, score on the
held-out one - the number that decides whether this is a scheme or an artifact). A basis is a
MODEL-side object, so the shipped file is fitted on all users; the held-out column is only the
generalization check.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/fit_fixedbasis.py \
      --npz scratchpad/quant_plan/corruption/state_probe_out/states_l2i8s4all.npz \
      --streams 1:note,2:card [--modes 16] [--out ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

K = 32


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


def loo(x, u, m):
    users = sorted({int(v) for v in u})
    if len(users) < 2:
        return None
    out = []
    for held in users:
        tr, te = x[u != held], x[u == held]
        if tr.shape[0] < 8 or te.shape[0] < 1:
            continue
        p, q = fit(tr, m)
        out.append(capture(te, p, q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--streams", default="1:note,2:card",
                    help="rank:stream, used when the dump key has no stream name")
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--m-report", default="2,3,4,6,8,12,16",
                    help="m values to report the captured energy for")
    args = ap.parse_args()

    p = Path(args.npz)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    by_rank = {}
    for pair in args.streams.split(","):
        r, _, s = pair.strip().partition(":")
        by_rank[int(r)] = s.strip()
    ms = [int(v) for v in args.m_report.split(",")]

    groups = {}          # (stream, layer) -> list of state arrays
    users = {}           # (stream, layer) -> user ids
    for i, key in enumerate(meta["keys"]):
        parts = key.split("|")
        stream = parts[0]
        rank = int(parts[1][1:])
        layer = int(parts[2][1:])
        if stream == "x":
            stream = by_rank.get(rank, f"rank{rank}")
        x = doc[f"a{i}"].astype(np.float32)
        un = f"u{i}"
        u = doc[un] if un in doc.files else np.full(x.shape[0], -1, dtype=np.int32)
        groups.setdefault((stream, layer), []).append(x)
        users.setdefault((stream, layer), []).append(u)

    arrays = {}
    report = {"npz": str(p), "modes": args.modes, "streams": by_rank, "by_key": {}}
    print(f"fitting M={args.modes} from {p.name} ({meta.get('total_states')} states)")
    head = f"{'key':>14} {'N':>6} | " + " ".join(f"m={m:<6}" for m in ms) + \
           "| LOO " + " ".join(f"m={m:<6}" for m in ms)
    print(head)
    print("-" * len(head))

    for (stream, layer), chunks in sorted(groups.items()):
        x = np.concatenate(chunks)
        u = np.concatenate(users[(stream, layer)])
        pv, qv = fit(x, args.modes)
        arrays[f"P__{stream}__L{layer}"] = pv
        arrays[f"Q__{stream}__L{layer}"] = qv
        ins, oos = {}, {}
        for m in ms:
            ins[m] = capture(x, pv[:, :m], qv[:, :m])
            l = loo(x, u, m)
            oos[m] = float(np.mean(l)) if l else None
        report["by_key"][f"{stream}__L{layer}"] = {
            "n": int(x.shape[0]), "users": sorted({int(v) for v in u}),
            "in_sample": ins, "loo": oos}
        print(f"{stream + '__L' + str(layer):>14} {x.shape[0]:>6} | "
              + " ".join(f"{ins[m]:.4f} " for m in ms) + "| LOO "
              + " ".join(f"{'-' if oos[m] is None else format(oos[m], '.4f')} " for m in ms))

    # pooled per-stream bases (one basis for every layer of the stream)
    for stream in sorted({s for s, _ in groups}):
        chunks = [c for (s, _), cc in groups.items() if s == stream for c in cc]
        us = [int(v) for (s, _), uu in users.items() if s == stream for arr in uu for v in arr]
        x = np.concatenate(chunks)
        u = np.asarray(us, dtype=np.int32)
        pv, qv = fit(x, args.modes)
        arrays[f"P__{stream}__ALL"] = pv
        arrays[f"Q__{stream}__ALL"] = qv
        ins, oos = {}, {}
        for m in ms:
            ins[m] = capture(x, pv[:, :m], qv[:, :m])
            l = loo(x, u, m)
            oos[m] = float(np.mean(l)) if l else None
        report["by_key"][f"{stream}__ALL"] = {"n": int(x.shape[0]), "in_sample": ins, "loo": oos}
        print(f"{stream + '__ALL':>14} {x.shape[0]:>6} | " + " ".join(f"{ins[m]:.4f} " for m in ms)
              + "| LOO " + " ".join(f"{'-' if oos[m] is None else format(oos[m], '.4f')} " for m in ms))

    out = Path(args.out) if args.out else p.parent / f"fixedbasis_{meta.get('tag')}.npz"
    np.savez_compressed(out, **arrays)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({len(arrays)} arrays) and {out.with_suffix('.json')}")
    print("point RWKV_FIXEDBASIS_PATH at the .npz; the QAT scope's rank slot is m.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
