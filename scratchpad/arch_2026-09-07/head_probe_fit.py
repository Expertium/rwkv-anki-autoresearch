"""Does the trunk output carry curve-relevant information the CURVE HEAD discards?

Motivation (headrank_screen.py, 2026-09-07): on the SAME 80-d trunk output (participation-ratio rank
10.0) the rating head's hidden spreads over rank 10.9 while the curve head's collapses to **2.4** --
3 dims carry 90% of its variance, for a 9-number (w,S,d)xN=3 output. So the forgetting curve is drawn
from a ~2.4-parameter family across every review. Waste, or the right inductive bias?

⚠ TWO EARLIER VERSIONS OF THIS SCREEN WERE INVALID, both recorded here so the third is trusted for
reasons rather than by default:
 1. A hand-rolled Newton solver diverged (400 steps, no line search) and returned held-out BCE of
    2.0-6.2 where a constant predictor gives ~0.30; the intercept column had also been standardised
    to all-zeros. It would have "proved" the head is fine -- the iter-67 un-diagnostic-probe failure.
 2. With a real solver the numbers were sane but the COMPARISON was confounded: the model trained on
    all these users, the probe was leave-one-user-out, so the model held an unfair advantage. Its own
    sanity check (const >= t-only >= x+t) also FAILED, correctly, and taught that the assumption was
    wrong: t-only transfers WORSE than a constant because the t->outcome relation shifts across
    users. "More features => lower held-out loss" is false under distribution shift.

THE CONFOUND-FREE TEST. Ask whether the head is a SUFFICIENT STATISTIC of its own input: take the
model's own output as a feature and see whether x adds anything on top of it.
    probe0 = logistic(logit p_model)                  -- recalibration only
    probe1 = logistic(logit p_model, t-basis)
    probe2 = logistic(logit p_model, t-basis, x)      -- can x improve on the head's own answer?
All leave-one-user-out, standardised on train rows only. Both arms are equally in-sample for the
model, so no arm has an advantage.
  SANITY: probe0 must land within 0.005 of the model's own BCE -- if a single monotone feature cannot
  reproduce the prediction it was built from, the fit is broken and no verdict is printed.
  VERDICT: probe2 - probe0. Below 0.002 => the head already extracts what the trunk offers about the
  curve, and a richer ahead head is NOT the lever. Above => it discards information, and the gap is
  an UPPER BOUND (the probe is unconstrained in t, which the deploy contract forbids).

Usage: head_probe_fit.py <dump.npz>
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def bce(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


d = np.load(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/arch_2026-09-07/head_dump.npz")
X, P, Y, T, U = d["x"].astype(np.float64), d["p"], d["y"], d["t"], d["u"]
lt = np.log1p(np.maximum(T, 1.0) / 86400.0)
TB = np.stack([lt, lt ** 2, np.exp(-lt), np.exp(-2 * lt)], 1)
LP = np.log(np.clip(P, EPS, 1 - EPS) / (1 - np.clip(P, EPS, 1 - EPS)))[:, None]
users = sorted(set(U.tolist()))
print(f"{len(X):,} labelled rows, trunk dim {X.shape[1]}, users {users}")

ARMS = {"probe0 (logit p)": LP,
        "probe1 (+t)": np.concatenate([LP, TB], 1),
        "probe2 (+t,+x)": np.concatenate([LP, TB, X], 1)}
res = {k: [] for k in ARMS}
res["MODEL"] = []
for uid in users:
    te, tr = U == uid, U != uid
    if te.sum() < 200 or tr.sum() < 1000:
        continue
    res["MODEL"].append(bce(P[te], Y[te]))
    line = f"  held-out {uid:>5}: n={te.sum():>6}  MODEL {res['MODEL'][-1]:.5f}"
    for name, F in ARMS.items():
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit((F[tr] - mu) / sd, Y[tr])
        res[name].append(bce(clf.predict_proba((F[te] - mu) / sd)[:, 1], Y[te]))
        line += f"  {name} {res[name][-1]:.5f}"
    print(line, flush=True)

m = {k: np.array(v) for k, v in res.items()}
print("\nby-user means: " + "  ".join(f"{k} {v.mean():.5f}" for k, v in m.items()))
cal = abs(m["probe0 (logit p)"].mean() - m["MODEL"].mean())
print(f"SANITY (probe0 reproduces the model within 0.005): {'PASS' if cal < 0.005 else 'FAIL'} (|d|={cal:.5f})")
if cal >= 0.005:
    print("NO VERDICT -- a monotone function of the model's own logit cannot reproduce it, so the fit")
    print("is unreliable (or the model is badly miscalibrated across users, which is its own finding).")
    raise SystemExit(3)
gap = m["probe0 (logit p)"] - m["probe2 (+t,+x)"]
gt = m["probe0 (logit p)"] - m["probe1 (+t)"]
print(f"\nwhat t alone adds beyond the head's output : {gt.mean():+.6f}")
print(f"what x + t add beyond the head's output    : {gap.mean():+.6f}  median {np.median(gap):+.6f}  "
      f"positive on {(gap > 0).sum()}/{len(gap)} users")
print("KILL LINE: below +0.002 => the curve head already extracts what the trunk offers about the")
print("curve; its rank-2.4 hidden is the right inductive bias, not waste, and a richer ahead head is")
print("NOT the lever. Above => the head discards information (UPPER BOUND: the probe may be")
print("non-monotone in t, which the deploy contract forbids).")
