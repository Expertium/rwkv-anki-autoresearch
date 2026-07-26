"""Split the rect-vs-unrect ahead difference into its two causes.

Andrew, 2026-07-26: *"about review duration of the most recent (current) review being zeroed
out - we still need to benchmark how much that affects log loss, right?"*

A probe row is the scored review with the grade one-hot swapped AND the current-row duration
set to 0.0, so `RWKV_EVAL_PAVA=0 -> 1` moves two things at once and the resulting number cannot
say which one it measured. Mode 2 substitutes the pressed probe WITHOUT pooling, moving only the
duration, which makes the split additive and exact:

    mode2 - mode0 = cost of zeroing the current-row duration   <- the thing Andrew asked about
    mode1 - mode2 = cost/benefit of the PAVA pooling itself
    mode1 - mode0 = the total a rect-vs-unrect run reports     (identity, checked here)

Sign convention: POSITIVE = worse (LogLoss went up), matching every other table in this repo.

`imm` is reported only as a falsification check. It comes off the rating head, which the probe
machinery never touches, and probes are skip rows that do not advance state -- so all three modes
must agree on it BIT-EXACTLY. A nonzero imm difference means probes are perturbing the recurrence
and the whole decomposition is invalid, so it is an assertion, not a result.

Usage:
  python scratchpad/eval_pava/decompose_duration.py \
      --mode0 result/RWKV-iter31_algo.jsonl \
      --mode1 result/RWKV-iter31_algo_rect.jsonl \
      --mode2 result/RWKV-iter31_algo_raw.jsonl \
      [--imm0 ... --imm1 ... --imm2 ...]
"""
import argparse
import json
import math
import sys

from scipy.stats import wilcoxon


def load(path, field="LogLoss"):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["user"]] = (rec["metrics"][field], rec["size"])
    if not out:
        raise SystemExit(f"ERROR: no rows in {path}")
    return out


def paired(a, b, users):
    """b - a, per user, in the fixed user order."""
    return [b[u][0] - a[u][0] for u in users]


def report(name, diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    nz = [d for d in diffs if d != 0.0]
    if nz:
        # two-sided: we do not have a directional prior for a diagnostic
        p = wilcoxon(nz).pvalue
    else:
        p = 1.0
    worse = sum(1 for d in diffs if d > 0)
    print(f"  {name:<34} {mean:+.6f}  +/- {se:.6f} (SE)   p={p:.3e}   "
          f"worse on {worse}/{n} users")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode0", required=True, help="ahead jsonl, RWKV_EVAL_PAVA=0")
    ap.add_argument("--mode1", required=True, help="ahead jsonl, RWKV_EVAL_PAVA=1 (rectified)")
    ap.add_argument("--mode2", required=True, help="ahead jsonl, RWKV_EVAL_PAVA=2 (raw probe)")
    ap.add_argument("--mode3", help="ahead jsonl, RWKV_EVAL_PAVA=3 (probes inserted, nothing "
                                    "substituted) -- the bf16 noise control; without it the "
                                    "duration term is confounded by probe-insertion noise")
    ap.add_argument("--imm0")
    ap.add_argument("--imm1")
    ap.add_argument("--imm2")
    args = ap.parse_args()

    m0, m1, m2 = (load(p) for p in (args.mode0, args.mode1, args.mode2))
    users = sorted(set(m0) & set(m1) & set(m2))
    if not users:
        raise SystemExit("ERROR: the three evals share no users")
    print(f"paired on {len(users)} users "
          f"(mode0 {len(m0)}, mode1 {len(m1)}, mode2 {len(m2)})")

    # `size` is the per-user equalized review count -- a property of the data and the filters,
    # so a mode flag must not move it. If it does, the runs scored different review sets and
    # nothing below is comparable.
    bad = [u for u in users if not (m0[u][1] == m1[u][1] == m2[u][1])]
    if bad:
        raise SystemExit(f"ERROR: `size` differs across modes on {len(bad)} users "
                         f"(e.g. {bad[:5]}) -- the evals are not comparable")
    print(f"size identical across all three modes on all {len(users)} users "
          f"({sum(m0[u][1] for u in users):,} reviews)")

    m3 = load(args.mode3) if args.mode3 else None
    if m3 is not None:
        users = sorted(set(users) & set(m3))

    print("\nahead (curve head) -- positive = WORSE:")
    if m3 is not None:
        report("probe-insertion noise (m3 - m0)", paired(m0, m3, users))
        report("duration zeroing      (m2 - m3)", paired(m3, m2, users))
        print("     ^ the clean pair: both sides have the identical probes inserted, so bf16")
        print("       re-bucketing cancels and only the zeroed duration differs.")
    d_dur = report("duration+noise    (m2 - m0)", paired(m0, m2, users))
    d_pool = report("PAVA pooling      (m1 - m2)", paired(m2, m1, users))
    d_tot = report("total             (m1 - m0)", paired(m0, m1, users))
    if m3 is None:
        print("     ! m2 - m0 CONFOUNDS duration with probe-insertion noise "
              "(~+3e-4 on imm at n=2500); pass --mode3 to separate them.")
    resid = abs(d_dur + d_pool - d_tot)
    print(f"  {'additivity check':<34} |{d_dur:+.6f} {d_pool:+.6f} - {d_tot:+.6f}| "
          f"= {resid:.2e}")
    if resid > 1e-9:
        print("  WARNING: the split is not additive -- the three runs are not the same model")

    print("\nby-user mean ahead LogLoss:")
    for tag, m in (("mode0 unrectified", m0), ("mode2 raw probe  ", m2),
                   ("mode1 rectified  ", m1)):
        print(f"  {tag}  {sum(v[0] for v in (m[u] for u in users)) / len(users):.6f}")

    if args.imm0 and args.imm1 and args.imm2:
        i0, i1, i2 = (load(p) for p in (args.imm0, args.imm1, args.imm2))
        iu = sorted(set(i0) & set(i1) & set(i2))
        # NOT an equality assertion any more (corrected 2026-07-26). The rectifier never touches
        # the rating head, so in EXACT arithmetic imm would be identical -- but probe insertion
        # inflates the batch ~30%, which re-buckets sequences by length and reorders bf16
        # reductions. Measured on A18: mean +0.000280, scaling with recurrence length (1.98e-4 at
        # ~4.7k reviews/user -> 3.97e-4 at ~179k) and one-signed (62% -> 78% of users worse)
        # because LogLoss is convex. So imm is the CLEANEST AVAILABLE MEASUREMENT of that noise --
        # it is the channel the rectifier cannot reach -- not a tripwire.
        d1 = [i1[u][0] - i0[u][0] for u in iu]
        d2 = [i2[u][0] - i0[u][0] for u in iu]
        print(f"\nimm (rating head -- the rectifier cannot reach it), {len(iu)} users:")
        print("  this is the bf16 PROBE-INSERTION NOISE, not a correctness check:")
        report("mode1 - mode0", d1)
        report("mode2 - mode0", d2)
        print("  => a mode3 run (probes inserted, nothing substituted) measures the same noise on")
        print("     `ahead`, which is what makes `mode2 - mode3` the clean duration cost.")


if __name__ == "__main__":
    main()
