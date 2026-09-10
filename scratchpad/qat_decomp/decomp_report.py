"""Report a QAT-tax decomposition against scratchpad/qat_decomp/PREREG.md.

On users 5001-5300, all arms evaluate the SAME QAT checkpoint (rectified VAL metric):
  T = deploy  - control   the whole tax (control = the full-precision twin, arm 1 = w10plain)
  S = deploy  - noshift   what shift quantization costs the adapted model
  C = deploy  - nowcb     what the WKV codebook + WKV norm quant cost it
  R = T - S - C           the rest: rank-1 truncation, the weights' drift under QAT, interactions.
                          No arm can isolate it -- removing rank-1 is structural (cell 3).
A component that is significantly NEGATIVE means the model does better WITH that quantizer than
without it -- co-adaptation, and then the difference is not a cost; the report says so instead of
printing a share.

Usage: python decomp_report.py <tag> [run_dir] [--ctrl w10plain]
"""
import json
import math
import os
import sys

from scipy.stats import wilcoxon

args = [a for a in sys.argv[1:] if not a.startswith("--")]
TAG = args[0]
RUNDIR = args[1] if len(args) > 1 else ""
CTRL = sys.argv[sys.argv.index("--ctrl") + 1] if "--ctrl" in sys.argv else "w10plain"
LO, HI = 5001, 5300
RES = os.environ.get("DECOMP_RESULT_DIR", "result")   # tests point this at a scratch dir
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}
# bit-split screen (PREREG P4): reconstruction-error changes measured 2026-09-10 on ws10's corpus
WKV_B12_LO, WKV_B12_HI, SHIFT_M2B8 = 0.066, 0.146, 0.11


def load(p):
    if not os.path.exists(p):
        return None
    return {r["user"]: r["metrics"]["LogLoss"] for r in
            (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}


def stats(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) if n > 1 else float("nan")
    nz = [x for x in diffs if x != 0.0]
    p2 = wilcoxon(nz).pvalue if len(nz) >= 10 else float("nan")
    p_neg = wilcoxon(nz, alternative="less").pvalue if len(nz) >= 10 else float("nan")
    return m, sd / math.sqrt(n), p2, p_neg


def main():
    out = {}
    for mode in ("ahead", "imm"):
        dep_all = load(f"{RES}/{PRE[mode]}{TAG}.jsonl")
        ctrl_all = load(f"{RES}/{PRE[mode]}{CTRL}.jsonl")
        if dep_all is None or ctrl_all is None:
            print(f"{mode}: deploy or control result missing -- nothing to report")
            return 1
        arms = {}
        for arm in ("noshift", "nowcb"):
            bad = RUNDIR and os.path.exists(os.path.join(RUNDIR, f"d{arm}_BANNERS_WRONG.txt"))
            a = load(f"{RES}/{PRE[mode]}{TAG}d{arm}.jsonl")
            arms[arm] = None if (bad or a is None) else a
            if bad:
                print(f"{mode}: arm {arm} EXCLUDED -- its shard log banners were wrong")
        pool = [dep_all, ctrl_all] + [a for a in arms.values() if a is not None]
        users = sorted(u for u in range(LO, HI + 1)
                       if all(u in d and math.isfinite(d[u]) for d in pool))
        full_users = sorted(u for u in dep_all if u in ctrl_all
                            and math.isfinite(dep_all[u]) and math.isfinite(ctrl_all[u]))
        T_full = sum(dep_all[u] - ctrl_all[u] for u in full_users) / len(full_users)
        res = {"n": len(users), "T_full": T_full, "n_full": len(full_users)}
        res["T"] = stats([dep_all[u] - ctrl_all[u] for u in users])
        for arm, key in (("noshift", "S"), ("nowcb", "C")):
            a = arms[arm]
            res[key] = None if a is None else stats([dep_all[u] - a[u] for u in users])
        out[mode] = res

    print(f"QAT-TAX DECOMPOSITION  tag={TAG}  control={CTRL}  users {LO}-{HI}")
    print("positive = that part costs LogLoss; SE is the paired standard error\n")
    print(f"{'part':34s} {'ahead':>22s} {'imm':>22s}")
    lab = {"T": "T  whole tax (deploy - control)", "S": "S  shift quantization",
           "C": "C  WKV codebook + norm quant"}
    for key in ("T", "S", "C"):
        row = []
        for mode in ("ahead", "imm"):
            s = out[mode][key]
            row.append("n/a" if s is None else f"{s[0]:+.6f} +/- {s[1]:.6f}")
        print(f"{lab[key]:34s} {row[0]:>22s} {row[1]:>22s}")
    for mode in ("ahead", "imm"):
        r = out[mode]
        if r["S"] is not None and r["C"] is not None:
            r["R"] = r["T"][0] - r["S"][0] - r["C"][0]
    print(f"{'R  rank-1 + drift + interaction':34s} "
          + " ".join(f"{('n/a' if 'R' not in out[m] else format(out[m]['R'], '+.6f')):>22s}"
                     for m in ("ahead", "imm")))
    print(f"\nn = {out['ahead']['n']} paired users; the same tax on all {out['ahead']['n_full']} users: "
          f"ahead {out['ahead']['T_full']:+.6f}  imm {out['imm']['T_full']:+.6f}")

    print("\nSIGN CHECK (a significantly negative part = co-adaptation, not a cost)")
    separable = True
    for mode in ("ahead", "imm"):
        for key in ("S", "C"):
            s = out[mode][key]
            if s is None:
                continue
            neg = s[0] < -0.0003 and s[3] < 0.01
            separable &= not neg
            print(f"  {mode:5s} {key}: mean {s[0]:+.6f}  p(two-sided) {s[2]:.2e}  p(arm better) "
                  f"{s[3]:.2e}{'   CO-ADAPTED' if neg else ''}")

    print("\nSHARES OF THE TAX (PREREG P2: C >= 50%, P3: S <= 20%)")
    for mode in ("ahead", "imm"):
        r = out[mode]
        T = r["T"][0]
        if T <= 0 or r["S"] is None or r["C"] is None:
            print(f"  {mode}: not computable (T {T:+.6f}, or a part is missing)")
            continue
        print(f"  {mode:5s} C {r['C'][0] / T:6.1%}   S {r['S'][0] / T:6.1%}   R {r['R'] / T:6.1%}")

    print("\nBIT-SPLIT SCREEN (PREREG P4; crude linear map from reconstruction error)")
    verdicts = []
    for mode in ("ahead", "imm"):
        r = out[mode]
        if r["S"] is None or r["C"] is None:
            print(f"  {mode}: n/a")
            verdicts.append(None)
            continue
        C, S = r["C"][0], r["S"][0]
        lo, hi = WKV_B12_LO * C - SHIFT_M2B8 * S, WKV_B12_HI * C - SHIFT_M2B8 * S
        print(f"  {mode:5s} predicted gain {lo:+.6f} (WKV -6.6%) .. {hi:+.6f} (WKV -14.6%)")
        verdicts.append((lo, hi))
    if None in verdicts or not separable:
        print("  => NOT DECIDABLE from this decomposition")
    elif all(v[0] >= 0.0001 for v in verdicts):
        print("  => QUEUE the bit-split A/B (conservative gain >= 0.0001 in both modes)")
    elif all(v[1] >= 0.0001 for v in verdicts):
        print("  => MARGINAL: only the optimistic map clears 0.0001 in both modes -- report, do not queue")
    else:
        print("  => DEAD: even the optimistic map is under 0.0001 in a mode")

    print("\nDETERMINISM (10-user probe vs the same users in the full run; both use the deploy env)")
    for mode in ("ahead", "imm"):
        p, f = load(f"{RES}/{PRE[mode]}{TAG}p.jsonl"), load(f"{RES}/{PRE[mode]}{TAG}.jsonl")
        if p and f:
            common = [u for u in p if u in f]
            mx = max(abs(p[u] - f[u]) for u in common) if common else float("nan")
            print(f"  {mode:5s} max |probe - full| over {len(common)} users = {mx:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
