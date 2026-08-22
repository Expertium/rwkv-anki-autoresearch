"""Choose the users to replay, stratified by collection size.

WHY 5001-7500 AND NOT ANY 2500 USERS. The RWKV champion was trained on users 1-5000, so
its intervals on those users would be fitted, not predicted, while FSRS-7's per-user
parameters are fitted for EVERY user by construction. Restricting to the VAL half is what
keeps the comparison from being rigged in RWKV's favour. (7501-10000 is the untouched TEST
half; CLAUDE.md's live rules reserve it for track close, and this is a characterisation,
not a gate, so VAL is the right side to spend.)

WHY STRATIFIED AND NOT RANDOM. Collection size spans nearly three orders of magnitude
(5th percentile 6.0k reviews, 99th 520k). A uniform sample is a sample of small
collections; a review-weighted one is a sample of two or three huge users. Strata make the
size dependence VISIBLE instead of averaging it away -- and whether the workload ratio
depends on collection size is itself worth knowing.

Cost is the reason the strata are not equal-sized: the RWKV arm runs at ~20 reviews/s on
one thread, so a 240k-review user is four hours by itself. Phase 1 takes the three
smallest strata (about 3 hours on two workers); the large strata are phase 2, run only if
phase 1's ratios look size-dependent enough to need them.

Usage: .venv/Scripts/python.exe scratchpad/workload/select_users.py <phase> [out.json]
"""
import sys
import json
from pathlib import Path

import numpy as np

ROWS_FILE = Path("scratchpad/workload/user_rows_5001_7500.json")
PARAM_FILE = Path(r"C:\Users\Andrew\srs-benchmark\result\FSRS-7-short-secs.jsonl")

# (low, high, n_users) -- review-count bands, inclusive low / exclusive high
PHASES = {
    "1": [(5000, 10000, 8), (10000, 20000, 8), (20000, 40000, 8)],
    "2": [(40000, 80000, 6), (80000, 160000, 4), (160000, 320000, 2)],
    # Phase 3 exists because of what phase 1 MEASURED, not as a pre-planned extension.
    # The per-user ratio came out with a consistent direction but a spread of 0.20 to 2.59
    # and a sign test that only reached p=0.043 at the two lowest DR levels -- n=25 cannot
    # resolve it. Meanwhile the ratio's dependence on collection size was FLAT (Spearman
    # rho -0.05..-0.31 across DR levels), so phase 2's twelve giant users would cost 9 h to
    # add twelve data points, while the same wall clock buys ~40 small ones. Statistical
    # power per CPU-hour is what is scarce here, and small users are ~7 min each.
    "3": [(5000, 12000, 24), (12000, 25000, 16)],
}
SEED = 20260821


def users_with_params():
    have = set()
    with open(PARAM_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                have.add(json.loads(line)["user"])
    return have


def main():
    phase = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "scratchpad/workload/users_phase%s.json" % phase)
    rows = {int(k): v for k, v in json.loads(ROWS_FILE.read_text()).items()}
    have = users_with_params()
    rng = np.random.default_rng(SEED)

    # never re-pick a user an earlier phase already replayed: a duplicate would be silently
    # skipped by the resume logic and quietly shrink the phase below its stated size
    already = set()
    for f in sorted(Path("scratchpad/workload").glob("users_phase*.json")):
        if f.name == out.name:
            continue
        already |= set(json.loads(f.read_text())["users"])
    already |= {int(p.stem.split("_u")[1])
                for p in Path("scratchpad/workload/out").glob("rwkv_u*.parquet")}

    picked, report = [], []
    for lo, hi, n in PHASES[phase]:
        pool = sorted(u for u, r in rows.items()
                      if lo <= r < hi and u in have and u not in already)
        if len(pool) <= n:
            sel = pool
        else:
            sel = sorted(rng.choice(pool, size=n, replace=False).tolist())
        picked += sel
        report.append({
            "band": [lo, hi], "pool": len(pool), "picked": sel,
            "reviews": int(sum(rows[u] for u in sel)),
        })

    total = sum(rows[u] for u in picked)
    out.write_text(json.dumps(
        {"phase": phase, "seed": SEED, "users": picked,
         "reviews": {str(u): rows[u] for u in picked},
         "total_reviews": int(total), "strata": report}, indent=1), encoding="utf-8")
    print("phase %s: %d users, %d reviews total" % (phase, len(picked), total))
    for r in report:
        print("  %7d-%-7d  pool %4d  picked %d  reviews %9d"
              % (r["band"][0], r["band"][1], r["pool"], len(r["picked"]), r["reviews"]))
    print("  estimated RWKV arm cost: %.1f h on 1 thread, %.1f h on 2"
          % (total / 20.2 / 3600, total / 20.2 / 3600 / 2))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
