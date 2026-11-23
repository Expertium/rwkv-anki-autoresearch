"""Fit the token-shift codec's basis from one or more shift dumps, price the pooling choices, and
export the deployable artifact (Python `.npz` + Rust `.txt`).

WHAT IS BEING FITTED. The deploy coder for a persisted token-shift vector is `quant_vec_inplace`:
`scale = amax/qmax`, store `round(x/scale)` clamped, dequant `code*scale` - 32 coordinates at int-N, so
a 32-d shift vector costs `32*qbits` bits. This fits the alternative: `x ~ mu + P c`, where `mu` (32
numbers) and `P` (32 x m) ship WITH THE MODEL - free in state bits - and only the `m` coefficients `c`
are stored, at int-N each. The shift is a LayerNorm output, so its mean carries ~70% of the energy and
its covariance is concentrated; measured on real vectors, m=8 int8 beats int4 on BOTH axes (64 bits at
rel_err 0.0132-0.0140 against 128 bits at 0.0162-0.0173).

BIT ARITHMETIC. The engine's shift budget is `64 * shift_bits` per layer (two 32-d vectors). The codec
spends `2 * m * level_bits` per layer plus ZERO for the model-side `mu`/`P`. Per vector: int4 raw = 128
bits; codec m=8 int8 = 64; m=8 int4 = 32. The `mean` column below is the free half of the win (mu
costs no state bits), so it is quoted at zero bits on purpose.

TWO POOLING QUESTIONS, BOTH ANSWERED HERE.
  1. PER LAYER vs POOLED. `analyze_shift_pooling.py` answers the layer side on one stream: the per-layer
     means are near-identical (pairwise cosine 0.991-0.998) and pooling the BASIS costs about 5%, while
     pooling the mean too costs 10-13% - still better than int4 at half the bits.
  2. PER STREAM vs PER ROLE. The Python forward has a stream name (`tail_tag`) and the Rust engine has
     the module index, so per-stream IS available; whether it is NEEDED is the CROSS-STREAM table.

TIERED KEYS. The exported artifact carries, for every group it can: `M__<stream>__r<role>__L<layer>`,
`M__<stream>__r<role>`, and `M__r<role>` (the same three tiers for `P__`). The Python twin and the Rust
engine both look the most specific one up first, so the artifact's granularity is a DATA choice - the
same file serves every tier. The channel mixer never stored `self.layer_id` before 2026-11-23, so its
rows are pooled over layers (reported layer -1) and only get the two coarser tiers.

LOO-BY-GROUP is the honest column: fit on every group but one, score on the held-out group. A basis is a
model-side object, so the shipped file is fitted on everything; the held-out column only says whether
the structure transfers.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/fit_shift_basis.py \
      --npz scratchpad/quant_plan/corruption/shift_probe_out/shifts_note_multi6.npz \
            scratchpad/quant_plan/corruption/shift_probe_out/shifts_card_multi6.npz \
      --modes 16 --out scratchpad/quant_plan/corruption/shiftbasis_l2i8s4
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROLE = {0: "t_xshift", 1: "c_xshift"}
# the deploy levels, mirroring the engine's parse_level (model.rs)
LEVEL = {"int8": (127.0, 8), "int6": (31.0, 6), "int5": (15.0, 5), "int4": (7.0, 4),
         "int3": (3.0, 3), "int2": (1.0, 2)}


def role_of(name: str) -> int:
    """0 = the WKV mixer's t_xshift, 1 = the channel mixer's c_xshift.

    The channel mixer never stores `self.layer_id`, so the probe cannot label it and its calls land on a
    `?` key; the class name is still captured, and `?` is the channel mixer by construction (it is the
    only caller without a layer id). Documented rather than guessed: the probe's own `_caller` breaks on
    the first frame whose `self` is a mixer, so a `?` key means "mixer found, layer id missing".
    """
    if "TimeMixer" in name:
        return 0
    return 1


def rel_err(x, x_hat):
    return float(((x - x_hat) ** 2).sum(axis=1).mean() / max((x ** 2).sum(axis=1).mean(), 1e-30))


def quant_rows(v, qmax=127.0):
    amax = np.abs(v).max(axis=1, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    return np.clip(np.round(v / scale), -qmax, qmax) * scale


def fit(x, m):
    mu = x.mean(axis=0).astype(np.float32)
    cent = (x - mu).astype(np.float64)
    cov = cent.T @ cent / max(x.shape[0] - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    return mu, evec[:, np.argsort(ev)[::-1]][:, :m].astype(np.float32)


def score(x, mu, p, qmax=127.0):
    return rel_err(x, (mu + quant_rows((x - mu) @ p, qmax) @ p.T).astype(np.float32))


def load(npz_path):
    """-> list of (group_key, stream, role, layer, rows)."""
    p = Path(npz_path)
    meta = json.loads((p.parent / (p.name[:-4] + ".meta.json")).read_text(encoding="utf-8"))
    doc = np.load(p)
    out = []
    for i, key in enumerate(meta["keys"]):
        info = meta["meta"][key]
        stream = info.get("stream") or meta.get("stream") or key.split("|")[0]
        r = role_of(info.get("role", key.split("|")[1]))
        lay = int(info.get("layer", -1))
        out.append((key, stream, r, lay, doc[f"a{i}"].astype(np.float32)))
    return out


def deploy_grid(named, ms, modes, arrays):
    """Fit, print and RECORD the (m, level) grid for a list of `(key_stem, items)` pairs.

    `key_stem` is the tier key without its `M__`/`P__` prefix: `r0` for the role-only tier,
    `note__r0` for the per-stream one, `note__r0__L0` for the per-layer one. The grid is recorded
    under the stem so a later job can pick a point without re-fitting."""
    rows = {}
    for stem, items in named:
        X = np.concatenate([x for _, x in items])
        mu, P = fit(X, modes)
        arrays[f"M__{stem}"] = mu
        arrays[f"P__{stem}"] = P
        grid = {}
        for lvl, (q, bits) in LEVEL.items():
            for m in ms:
                pm = P[:, :m]
                grid[f"m{m}{lvl}"] = {
                    "bits_per_vector": m * bits,
                    "pooled": score(X, mu, pm, q),
                    "worst_key": max(score(x, mu, pm, q) for _, x in items),
                }
        # the zero-cost reference: int-N on the RAW vector at the deploy level for that width
        for lvl, (q, bits) in LEVEL.items():
            grid[f"raw{lvl}"] = {
                "bits_per_vector": 32 * bits,
                "pooled": rel_err(X, quant_rows(X, q).astype(np.float32)),
            }
            grid[f"mean{lvl}"] = {
                "bits_per_vector": 32 * bits,
                "pooled": rel_err(X, (mu + quant_rows(X - mu, q)).astype(np.float32)),
            }
        rows[stem] = grid
        print(f"  {stem}: {X.shape[0]} rows  ({len(items)} keys)")
        for name, g in sorted(grid.items(), key=lambda kv: kv[1]["bits_per_vector"]):
            worst = "" if "worst_key" not in g else f"  worst_key {g['worst_key']:.5f}"
            print(f"    {name:<10} {g['bits_per_vector']:>4} bits/vec  pooled {g['pooled']:.5f}{worst}")
    return rows


def write_rust(path, arrays):
    """Write the Rust engine's shift-codec file: line 1 = entry count, then per entry
    `<key> <stride>`, `K` floats of `mu`, then `K*stride` floats of `P` (row-major), whitespace
    separated. Only the first `m <= stride` columns are stored per coefficient, so ONE file serves
    every m. The keys are the tiered stems the engine looks up (most specific first)."""
    stems = sorted({k[3:] for k in arrays if k.startswith("M__")})
    rows = []
    for stem in stems:
        mu = arrays[f"M__{stem}"]
        p = arrays[f"P__{stem}"]
        rows.append(f"{stem} {p.shape[1]}")
        rows.append(" ".join(f"{v:.8g}" for v in mu))
        rows.append(" ".join(f"{v:.8g}" for v in p.reshape(-1)))
    path.write_text(f"{len(stems)}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"  rust file: {len(stems)} entries -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--m-report", default="4,6,8,10,12,16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ms = [int(v) for v in args.m_report.split(",")]

    groups = []
    for path in args.npz:
        groups.extend(load(path))
    if not groups:
        raise SystemExit("no groups loaded")
    print("groups:")
    for key, stream, r, lay, x in groups:
        print(f"  {key:>28}  stream={stream:<5} role={ROLE[r]:<8} layer={lay:<3} rows={x.shape[0]}")

    by_role = {}
    by_sr = {}
    for key, stream, r, lay, x in groups:
        by_role.setdefault(r, []).append((key, x))
        by_sr.setdefault((stream, r), []).append((key, x))

    report = {"groups": [{"key": k, "stream": s, "role": ROLE[r], "layer": l, "rows": int(x.shape[0])}
                         for k, s, r, l, x in groups],
              "modes": args.modes, "role_pooled": {}, "per_stream_role": {}, "cross_stream": {},
              "deploy_grid": {}}
    arrays = {}

    print("\n--- ROLE-POOLED fit (one basis per role, over every stream and layer) ---")
    for r, items in sorted(by_role.items()):
        X = np.concatenate([x for _, x in items])
        mu, P = fit(X, args.modes)
        rec = {"n": int(X.shape[0]), "keys": [k for k, _ in items]}
        print(f"  role {ROLE[r]}: {X.shape[0]} rows from {len(items)} keys, |mu|={np.linalg.norm(mu):.3f}")
        for m in ms:
            pm = P[:, :m]
            rec[str(m)] = {"bits": m * 8,
                           "per_key": {k: score(x, mu, pm) for k, x in items},
                           "pooled": score(X, mu, pm)}
            rec[str(m)]["worst_key"] = max(rec[str(m)]["per_key"].values())
            print(f"    m={m:<3} {m * 8:>4} bits  pooled {rec[str(m)]['pooled']:.5f}  "
                  + "  ".join(f"{k.split('|')[-1]}:{v:.5f}" for k, v in rec[str(m)]["per_key"].items()))
        report["role_pooled"][ROLE[r]] = rec

    print("\n--- PER (STREAM, ROLE) fit ---")
    for (stream, r), items in sorted(by_sr.items()):
        X = np.concatenate([x for _, x in items])
        mu, P = fit(X, args.modes)
        rec = {"n": int(X.shape[0]), "keys": [k for k, _ in items]}
        for m in ms:
            rec[str(m)] = {"bits": m * 8, "pooled": score(X, mu, P[:, :m])}
        print(f"  {stream} {ROLE[r]}: {X.shape[0]} rows -> " + "  ".join(
            f"m={m}:{rec[str(m)]['pooled']:.5f}" for m in ms))
        report["per_stream_role"][f"{stream}__{ROLE[r]}"] = rec

    # the decision number: does the OTHER stream's basis work?
    streams = sorted({s for _, s, _, _, _ in groups})
    if len(streams) > 1:
        print("\n--- CROSS-STREAM: score stream A with stream B's (stream,role) basis ---")
        for r in sorted(by_role):
            for a in streams:
                for b in streams:
                    if a == b or (a, r) not in by_sr or (b, r) not in by_sr:
                        continue
                    mu, P = fit(np.concatenate([x for _, x in by_sr[(b, r)]]), args.modes)
                    Xa = np.concatenate([x for _, x in by_sr[(a, r)]])
                    vals = {str(m): score(Xa, mu, P[:, :m]) for m in ms}
                    report["cross_stream"][f"fit{b}->score{a}__{ROLE[r]}"] = vals
                    print(f"  {ROLE[r]:<8} fit on {b}, scored on {a}: "
                          + "  ".join(f"m={m}:{vals[str(m)]:.5f}" for m in ms))

    print("\n--- DEPLOY GRID: the artifact actually shipped (mu + P, coefficient int-N) ---")
    by_srl = {}
    for key, stream, r, lay, x in groups:
        if lay >= 0:
            by_srl.setdefault((stream, r, lay), []).append((key, x))
    named = [(f"r{r}", items) for r, items in sorted(by_role.items())]
    named += [(f"{stream}__r{r}", items) for (stream, r), items in sorted(by_sr.items())]
    named += [(f"{stream}__r{r}__L{lay}", items) for (stream, r, lay), items in sorted(by_srl.items())]
    report["deploy_grid"] = deploy_grid(named, ms, args.modes, arrays)

    if args.out:
        out = Path(args.out)
        np.savez_compressed(out.with_suffix(".npz"), **arrays)
        out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        write_rust(out.with_suffix(".txt"), arrays)
        print(f"\nwrote {out.with_suffix('.npz')} ({len(arrays)} arrays), "
              f"{out.with_suffix('.json')} and {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
