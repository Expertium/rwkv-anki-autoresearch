"""Price the RWKV-vs-FSRS disagreement properly: the BEST blend, not a 50/50 one.

WHY A SECOND SCRIPT. cmp_fsrs.py's 50/50 oracle is the statistic the 2026-07-03 note used, and
it is only meaningful when the two models are of similar quality. FSRS-7 is 0.0163 by-user BCE
WORSE than realcyc here, so a 50/50 blend loses even if the disagreement is real information.
The question "how much is the out-of-family structure worth" needs the blend weight FITTED.

THREE ESTIMATES, increasing in flexibility, all reported by-user (the gate's own metric):
  w*  in probability space  -- one parameter
  w*  in LOGIT space        -- one parameter, the natural space for a log-loss blend
  logistic stack on both logits + intercept -- three parameters

HONESTY. The one-parameter fits are in-sample; with 340k rows and one parameter the optimism is
negligible, and it is an UPPER bound, which is what a ceiling argument needs. The 3-parameter
stack is reported BOTH in-sample and leave-one-user-out, because 4 users is few enough that
cross-user transfer is the real question (the arch screen already showed "more features => lower
held-out loss" is false here).
"""

from pathlib import Path

import numpy as np

HERE = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad")
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def by_user(u, y, p):
    p = np.clip(p, EPS, 1 - EPS)
    v = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.mean([v[u == uu].mean() for uu in np.unique(u)]))


def load():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cmp_fsrs", HERE / "fsrs7" / "cmp_fsrs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = load()
    # Re-use cmp_fsrs's loading + remap + intersection by calling its main pieces inline.
    maps = {}
    for a, arm in (("A", "realcyc"), ("B", "d128")):
        f = HERE / "fsrs7" / f"labelmap_{a}.npz"
        mm = np.load(f)
        maps[arm] = dict(zip((mm["u"] * 10_000_000 + mm["rth"]).tolist(), mm["lrth"].tolist()))
    loaded = {}
    for name, path in m.ARMS.items():
        if not path.exists():
            continue
        d = np.load(path)
        key = d["u"].astype(np.int64) * 10_000_000 + d["rth"].astype(np.int64)
        if name in maps:
            mp = maps[name]
            key = d["u"].astype(np.int64) * 10_000_000 + np.array(
                [mp[k] for k in key.tolist()], dtype=np.int64)
        loaded[name] = {"key": key, "p": d["p"].astype(np.float64),
                        "y": d["y"].astype(np.float64), "u": d["u"].astype(np.int64)}
    names = list(loaded)
    common = loaded[names[0]]["key"]
    for n in names[1:]:
        common = np.intersect1d(common, loaded[n]["key"])
    A = {}
    for n in names:
        d = loaded[n]
        order = np.argsort(d["key"])
        sel = order[np.searchsorted(d["key"], common, sorter=order)]
        A[n] = {"p": d["p"][sel], "y": d["y"][sel], "u": d["u"][sel]}
    y, u = A[names[0]]["y"], A[names[0]]["u"]
    print(f"rows={len(y):,}  users={sorted(set(u.tolist()))}\n")

    base = "realcyc"
    pb = A[base]["p"]
    b0 = by_user(u, y, pb)
    print(f"{base} alone (by-user BCE): {b0:.5f}\n")
    print(f"{'partner':<12}{'alone':>9}{'w*prob':>9}{'gain':>10}{'w*logit':>10}{'gain':>10}"
          f"{'stack':>10}{'stack LOU':>11}")
    ws = np.linspace(0, 1, 201)
    for n in names:
        if n == base:
            continue
        pa = A[n]["p"]
        alone = by_user(u, y, pa)
        gp = [by_user(u, y, (1 - w) * pb + w * pa) for w in ws]
        ip = int(np.argmin(gp))
        la, lb = logit(pa), logit(pb)
        gl = [by_user(u, y, 1 / (1 + np.exp(-((1 - w) * lb + w * la)))) for w in ws]
        il = int(np.argmin(gl))

        # 3-parameter logistic stack, in-sample and leave-one-user-out.
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([lb, la])
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        stack_in = by_user(u, y, clf.predict_proba(X)[:, 1])
        pred = np.empty_like(y)
        for uu in np.unique(u):
            te = u == uu
            c = LogisticRegression(max_iter=1000).fit(X[~te], y[~te])
            pred[te] = c.predict_proba(X[te])[:, 1]
        stack_lou = by_user(u, y, pred)
        print(f"{n:<12}{alone:>9.5f}{ws[ip]:>9.3f}{b0 - gp[ip]:>+10.5f}"
              f"{ws[il]:>10.3f}{b0 - gl[il]:>+10.5f}{b0 - stack_in:>+10.5f}"
              f"{b0 - stack_lou:>+11.5f}")


if __name__ == "__main__":
    main()
