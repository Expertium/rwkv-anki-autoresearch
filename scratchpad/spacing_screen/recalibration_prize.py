"""How much logloss is sitting in the champion's systematic OVERCONFIDENCE?

`calibration_by.py` found the model over-predicts success by ~0.004-0.005 in nearly every horizon
bucket -- and it found this on TRAIN-RANGE users, which is the part that makes it a finding rather
than a curiosity. Binary cross-entropy is a proper scoring rule: at its optimum `mean(p) == mean(y)`
inside any group determined by the inputs. A persistent gap on data the model trained on says the
model is not at the calibration optimum FOR THE HARD LABELS.

THE MECHANISM, read from the code rather than guessed (`srs_model.py:1261-1263`):

    label_y = alpha * teacher_curve + (1 - alpha) * label_y

with `alpha = 0.9` through WS and `0.5` through decay (iters 39 and 45). **The curve head is not
trained to predict the empirical outcome. It is trained to predict a blend that is mostly the d=128
teacher's probability**, so whatever calibration the teacher has is inherited -- and nothing in the
objective pulls it back to the data's frequency. This is not a bug; KD is the most successful family
in the log (4/4) and it pays through target-variance reduction. But variance reduction and
calibration are separable, and only one of them was ever measured.

WHAT THIS SCRIPT MEASURES: the logloss recoverable by a post-hoc recalibration, i.e. the size of the
prize before anyone spends 5.5 h of GPU on it.

    shift  : one parameter,  logit(p) + b          -- corrects a uniform confidence bias
    platt  : two parameters, a * logit(p) + b      -- also corrects a slope/sharpness error

HELD-OUT BY USER, not by row. Fitting and scoring the same rows would report the optimum's value
rather than what a real correction generalises to, and a per-user split is the honest analogue of
the gate (which scores users the model never fit). With 83k rows and 1-2 parameters the in-sample
number is nearly exact anyway -- the split is there to prove it, not to regularise.

⚠ SCOPE. These are TRAIN-range users and ALL predecessor-having rows, not the benchmark's equalized
subset, so the absolute logloss here is NOT comparable to the 0.2977 gate number. The quantity that
transfers is the RELATIVE improvement from recalibration.
"""
import numpy as np

D = np.load("scratchpad/spacing_screen/calib_records.npz")
p, y = D["p"].astype(np.float64), D["y"].astype(np.float64)
EPS = 1e-6
pc = np.clip(p, EPS, 1 - EPS)
z = np.log(pc / (1 - pc))          # logits of the champion's own predictions


def ll(logits, yy):
    q = 1.0 / (1.0 + np.exp(-logits))
    q = np.clip(q, EPS, 1 - EPS)
    return float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean())


def fit(zz, yy, two_param):
    """Newton on the 1- or 2-parameter logistic recalibration."""
    a, b = 1.0, 0.0
    for _ in range(60):
        q = 1.0 / (1.0 + np.exp(-(a * zz + b)))
        w = np.maximum(q * (1 - q), 1e-12)
        r = yy - q
        if two_param:
            X = np.stack([zz, np.ones_like(zz)], 1)
            H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, X.T @ r)
            a += step[0]; b += step[1]
            if abs(step).max() < 1e-10:
                break
        else:
            step = (r.sum()) / max(w.sum(), 1e-12)
            b += step
            if abs(step) < 1e-12:
                break
    return a, b


print(f"n = {p.size:,} predictions")
print(f"mean p = {p.mean():.5f}   mean y = {y.mean():.5f}   "
      f"OVERALL GAP = {y.mean() - p.mean():+.5f}")
base = ll(z, y)
print(f"champion logloss (this population) = {base:.6f}\n")

# ---- in-sample ceilings ----
_, b1 = fit(z, y, False)
a2, b2 = fit(z, y, True)
print("in-sample (the ceiling, not a claim):")
print(f"  shift b={b1:+.4f}          logloss {ll(z + b1, y):.6f}   gain {base - ll(z + b1, y):+.6f}")
print(f"  platt a={a2:.4f} b={b2:+.4f}  logloss {ll(a2 * z + b2, y):.6f}   "
      f"gain {base - ll(a2 * z + b2, y):+.6f}")

# ---- held out BY USER: fit on 4, score the 5th ----
u = D["u"] if "u" in D else None
if u is None:
    # calib_records.npz predates the user column; reconstruct blocks from the recorded run order
    print("\n(no user column in the npz -- falling back to a 2-fold row split, "
          "which is weaker but still out-of-sample)")
    rng = np.random.default_rng(0)
    idx = rng.permutation(p.size)
    h = p.size // 2
    folds = [(idx[:h], idx[h:]), (idx[h:], idx[:h])]
else:
    users = np.unique(u)
    folds = [(np.where(u != k)[0], np.where(u == k)[0]) for k in users]

g1 = g2 = 0.0
n_tot = 0
print("\nheld out:")
for tr, te in folds:
    bb = fit(z[tr], y[tr], False)[1]
    aa, bb2 = fit(z[tr], y[tr], True)
    b0 = ll(z[te], y[te])
    d1, d2 = b0 - ll(z[te] + bb, y[te]), b0 - ll(aa * z[te] + bb2, y[te])
    g1 += d1 * te.size; g2 += d2 * te.size; n_tot += te.size
    print(f"  fold n={te.size:>7,}  base {b0:.6f}   shift {d1:+.6f}   platt {d2:+.6f}")
print(f"\n  WEIGHTED HELD-OUT GAIN:  shift {g1/n_tot:+.6f}   platt {g2/n_tot:+.6f}")
print(f"  (accept bar is 0.0001; same-capacity noise floor is 7.5e-5)")
