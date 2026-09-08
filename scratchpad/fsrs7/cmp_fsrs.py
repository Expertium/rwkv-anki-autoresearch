"""Pairwise residual-decorrelation table over every dumped arm.

THE QUESTION. realcyc and the pretrained d=128 model have a residual correlation of 0.9957 on
ahead -- two RWKV-7 models, 5x apart in parameters and on different feature layouts, getting the
same reviews wrong. That is what a blind spot SHARED BY THE FAMILY looks like, and it is also
what TRUE NOISE looks like; the 2026-07-03 entropy-floor note says exactly that and could not
separate them. FSRS is the out-of-family arm that can.

WHY EVERY PAIR IS RECOMPUTED ON ONE COMMON SUBSET. FSRS scores its TimeSeriesSplit test folds;
our dumps carry every ahead row. Comparing an FSRS pair measured on 200k rows against an RWKV
pair measured on 410k rows would confound the contrast with the row set. So this intersects ALL
arms first and reports every pair on the identical rows -- including the RWKV-vs-RWKV reference,
whose number therefore need not equal the 0.9957 measured on the wider set.

ALIGNMENT IS PROVEN, NOT ASSUMED. The arms read different datasets (-id vs published) and share
only `review_th`. The script requires the LABELS to agree on the intersection; a misalignment
shows up as label disagreement rather than as a plausible-looking correlation.

READING IT. `resid corr` is the headline. Compare:
  RWKV vs RWKV      the in-family baseline on these rows
  FSRS-6 vs FSRS-7  the other family's own version-bump floor
  RWKV vs FSRS-7    the question
If the cross-family number is not materially below both, the two families disagree no more than
one version does, and the ahead residual is noise. `ens gain` prices any decorrelation in the
metric's own units: what a 50/50 oracle ensemble would buy over the better member.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad")
ARMS = {
    "realcyc": HERE / "arch_2026-09-07" / "decor_A.npz",
    "d128": HERE / "arch_2026-09-07" / "decor_B.npz",
    "FSRS-7": HERE / "fsrs7" / "fsrs_fsrs7.npz",
    "FSRS-6": HERE / "fsrs7" / "fsrs_fsrs6.npz",
    "FSRS-7def": HERE / "fsrs7" / "fsrs_fsrs7def.npz",
}
EPS = 1e-9


def bce(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def by_user_bce(u, y, p):
    v = bce(y, p)
    return float(np.mean([v[u == uu].mean() for uu in np.unique(u)]))


def main():
    # Our dumps key on review_th (the row the prediction is made FROM); FSRS keys on the row
    # being PREDICTED. Remap ours through label_review_th so both mean the same review.
    maps = {}
    for a, arm in (("A", "realcyc"), ("B", "d128")):
        f = HERE / "fsrs7" / f"labelmap_{a}.npz"
        if f.exists():
            m = np.load(f)
            maps[arm] = dict(zip((m["u"] * 10_000_000 + m["rth"]).tolist(), m["lrth"].tolist()))

    loaded = {}
    for name, path in ARMS.items():
        if not path.exists():
            print(f"[skip] {name}: {path.name} not present")
            continue
        d = np.load(path)
        key = d["u"].astype(np.int64) * 10_000_000 + d["rth"].astype(np.int64)
        if name in maps:
            mp = maps[name]
            assert all(k in mp for k in key.tolist()), f"{name}: label map does not cover the dump"
            key = d["u"].astype(np.int64) * 10_000_000 + np.array(
                [mp[k] for k in key.tolist()], dtype=np.int64)
        assert len(np.unique(key)) == len(key), f"{name}: duplicate (user, review_th)"
        loaded[name] = {"key": key, "p": d["p"].astype(np.float64),
                        "y": d["y"].astype(np.float64), "u": d["u"].astype(np.int64)}
        print(f"[load] {name:10s} rows={len(key):>8,}  users={len(np.unique(d['u']))}")
    names = list(loaded)
    if len(names) < 2:
        sys.exit("need at least two arms")

    common = loaded[names[0]]["key"]
    for n in names[1:]:
        common = np.intersect1d(common, loaded[n]["key"], assume_unique=True)
    print(f"\ncommon rows across all {len(names)} arms: {len(common):,}")
    if len(common) == 0:
        sys.exit("empty intersection -- review_th does not align")

    A = {}
    for n in names:
        d = loaded[n]
        idx = np.searchsorted(d["key"], common, sorter=np.argsort(d["key"]))
        order = np.argsort(d["key"])
        sel = order[idx]
        A[n] = {"p": d["p"][sel], "y": d["y"][sel], "u": d["u"][sel]}

    y = A[names[0]]["y"]
    u = A[names[0]]["u"]
    for n in names[1:]:
        agree = float((A[n]["y"] == y).mean())
        print(f"  label agreement {names[0]} vs {n:10s}: {agree:.6f}")
        assert agree > 0.999, f"{n}: labels disagree -- review_th does not align"

    print(f"\n{'arm':<11}{'Brier':>10}{'BCE':>10}{'by-user BCE':>14}")
    for n in names:
        p = A[n]["p"]
        print(f"{n:<11}{np.mean((y - p) ** 2):>10.5f}{bce(y, p).mean():>10.5f}"
              f"{by_user_bce(u, y, p):>14.5f}")

    print(f"\n{'pair':<24}{'resid cov':>11}{'/BrierA':>9}{'/BrierB':>9}"
          f"{'resid corr':>12}{'pred corr':>11}{'ens gain':>10}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra, rb = y - A[a]["p"], y - A[b]["p"]
            cov = float(np.mean(ra * rb))
            ba, bb = float(np.mean(ra ** 2)), float(np.mean(rb ** 2))
            corr = float(np.corrcoef(ra, rb)[0, 1])
            pcorr = float(np.corrcoef(A[a]["p"], A[b]["p"])[0, 1])
            ens = by_user_bce(u, y, 0.5 * (A[a]["p"] + A[b]["p"]))
            best = min(by_user_bce(u, y, A[a]["p"]), by_user_bce(u, y, A[b]["p"]))
            print(f"{a + ' vs ' + b:<24}{cov:>11.5f}{cov / ba:>9.4f}{cov / bb:>9.4f}"
                  f"{corr:>12.4f}{pcorr:>11.4f}{best - ens:>+10.5f}")


if __name__ == "__main__":
    main()
