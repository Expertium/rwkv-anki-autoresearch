"""Task #18: estimate the dataset's irreducible LogLoss (Bayes floor) on users 1-100.

Method (design: research_5k_notes.md "Queued analysis"): both pretrained d=128 models were
trained on DISJOINT user halves and users 1-100 were seen by neither. With y = p* + noise and
(approximately) independent model errors around p*,
    E[(y - pA)(y - pB)] ~= E[p*(1-p*)]   (the irreducible BRIER score; correlated errors bias UP).
The LogLoss floor needs p*'s dispersion, which single Bernoulli draws can't reveal (mixture
non-identifiability), so one parametric step: within calibration bins, assume p* ~ Beta with
mean = the bin's observed outcome rate and variance = m(1-m) - E[p*(1-p*)|bin]; then
E[H(p*)] has a closed form via digamma. Report pooled + by-user (benchmark metric) versions,
with a user-level bootstrap CI, plus Andrew's constant-retention baselines.

Run AFTER run_floor_est.cmd finishes:  python optimization/entropy_floor.py
"""
import json
import os

import numpy as np
from scipy.special import digamma

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EPS = 1e-6
N_BINS = 20
N_BOOT = 1000


def load_mode(name_a, name_b):
    """-> list of per-user dicts with aligned pA, pB, y arrays."""
    def read(path):
        users = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                users[r["user"]] = r
        return users

    ua = read(os.path.join(ROOT, "raw", f"{name_a}.jsonl"))
    ub = read(os.path.join(ROOT, "raw", f"{name_b}.jsonl"))
    assert set(ua) == set(ub), "user sets differ between models"
    out = []
    for uid in sorted(ua):
        a, b = ua[uid], ub[uid]
        assert a["review_th"] == b["review_th"], f"review_th mismatch user {uid}"
        ya = np.asarray(a["y"], dtype=np.float64).ravel()
        yb = np.asarray(b["y"], dtype=np.float64).ravel()
        assert np.array_equal(ya, yb), f"labels differ user {uid}"
        out.append({
            "user": uid,
            "y": ya,
            "pA": np.clip(np.asarray(a["p"], dtype=np.float64).ravel(), EPS, 1 - EPS),
            "pB": np.clip(np.asarray(b["p"], dtype=np.float64).ravel(), EPS, 1 - EPS),
        })
    return out


def bin_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def beta_mean_entropy(m, v):
    """E[H(p*)] for p* ~ Beta(mean m, variance v); v<=0 -> point mass -> H(m)."""
    m = float(np.clip(m, EPS, 1 - EPS))
    vmax = m * (1 - m)
    if v <= 0:
        return float(bin_entropy(np.array([m]))[0])
    v = min(v, vmax * (1 - 1e-9))
    nu = vmax / v - 1
    al, be = m * nu, (1 - m) * nu
    s = al + be
    e_plnp = (al / s) * (digamma(al + 1) - digamma(s + 1))
    e_qlnq = (be / s) * (digamma(be + 1) - digamma(s + 1))
    return float(-(e_plnp + e_qlnq))


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def floor_by_bins(users):
    """Beta-translated floor. Returns (pooled_floor, by_user_floor, per_user_floors, cross)."""
    sel = users
    y = np.concatenate([u["y"] for u in sel])
    pa = np.concatenate([u["pA"] for u in sel])
    pb = np.concatenate([u["pB"] for u in sel])
    pm = (pa + pb) / 2
    qs = np.quantile(pm, np.linspace(0, 1, N_BINS + 1))
    qs[0], qs[-1] = 0.0, 1.0 + 1e-12
    which = np.searchsorted(qs, pm, side="right") - 1
    which = np.clip(which, 0, N_BINS - 1)
    h_bin = np.zeros(N_BINS)
    for b in range(N_BINS):
        mask = which == b
        if not mask.any():
            continue
        m = y[mask].mean()
        irr_brier = np.mean((y[mask] - pa[mask]) * (y[mask] - pb[mask]))
        v = m * (1 - m) - irr_brier          # Var(p*|bin) implied by the covariance estimator
        h_bin[b] = beta_mean_entropy(m, v)
    pooled = float(np.mean(h_bin[which]))
    # by-user: each review contributes its bin's E[H|bin]; user mean; mean over users
    per_user = []
    off = 0
    for u in sel:
        n = len(u["y"])
        per_user.append(float(np.mean(h_bin[which[off:off + n]])))
        off += n
    cross = float(np.mean((y - pa) * (y - pb)))
    return pooled, float(np.mean(per_user)), np.asarray(per_user), cross


def analyze(mode, name_a, name_b):
    users = load_mode(name_a, name_b)
    y = np.concatenate([u["y"] for u in users])
    pa = np.concatenate([u["pA"] for u in users])
    pb = np.concatenate([u["pB"] for u in users])

    ll_a = float(np.mean([logloss(u["y"], u["pA"]) for u in users]))   # by-user, benchmark-style
    ll_b = float(np.mean([logloss(u["y"], u["pB"]) for u in users]))
    brier_a, brier_b = float(np.mean((y - pa) ** 2)), float(np.mean((y - pb) ** 2))

    pbar = y.mean()
    const_global = float(np.mean([logloss(u["y"], np.full_like(u["y"], pbar)) for u in users]))
    const_peruser = float(np.mean([logloss(u["y"], np.full_like(u["y"],
                                   np.clip(u["y"].mean(), EPS, 1 - EPS))) for u in users]))

    pooled_floor, byuser_floor, per_user_floors, cross = floor_by_bins(users)

    # user-level bootstrap over the precomputed per-user floors (bin mapping held fixed --
    # the naive full refit was ~1.5 s/rep x 1000 x 2 modes; this is the standard cheap CI)
    rng = np.random.default_rng(1234)
    boots = [np.mean(per_user_floors[rng.integers(0, len(per_user_floors),
                                                  len(per_user_floors))])
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n=== {mode} ===  ({len(users)} users, {len(y):,} equalized reviews, "
          f"mean retention {pbar:.4f})")
    print(f"constant p=global retention      : logloss {const_global:.4f}  (H(pbar)={bin_entropy(np.array([pbar]))[0]:.4f})")
    print(f"constant p=per-user retention    : logloss {const_peruser:.4f}  (by-user mean)")
    print(f"model A (tr 101-4999)  by-user   : logloss {ll_a:.4f}   brier {brier_a:.4f}")
    print(f"model B (tr 5000-10000) by-user  : logloss {ll_b:.4f}   brier {brier_b:.4f}")
    print(f"cross-model residual covariance  : {cross:.4f}   (irreducible-BRIER estimate, biased UP)")
    print(f"Beta-translated LogLoss FLOOR    : by-user {byuser_floor:.4f}  "
          f"[95% CI {lo:.4f}, {hi:.4f}]   pooled {pooled_floor:.4f}")
    print(f"  -> headroom vs model A          : {ll_a - byuser_floor:+.4f} by-user logloss")


def main():
    os.chdir(ROOT)
    analyze("AHEAD (next-review prediction)", "RWKV-floorA", "RWKV-floorB")
    analyze("IMM (immediate / benchmark-primary)", "RWKV-P-floorA", "RWKV-P-floorB")
    print("\nCaveats: correlated model errors bias the covariance (and thus the floor) UP -> this is")
    print("an upper-leaning estimate of the true floor; 100 users; Beta shape is an assumption.")


if __name__ == "__main__":
    main()
