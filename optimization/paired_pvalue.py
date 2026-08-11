"""Paired per-user significance test: candidate vs champion on the SAME fixed eval users.

The 5k acceptance gate (Andrew 2026-07-08) requires, in addition to the >=0.0003-both-modes
improvement, that the candidate's per-user LogLoss improvement over the current champion is
statistically real: one-sided Wilcoxon signed-rank on the per-user diffs must give
p < 0.0001 in BOTH modes (ahead AND imm).

Both models must have been evaluated on the identical user set (the fixed 5001-10000 eval),
so the diffs are exactly paired -- zero extra GPU cost, the data is already in the result jsonls.

Usage:
  python optimization/paired_pvalue.py \
      --cand-ahead result/RWKV-<cand>.jsonl --cand-imm result/RWKV-P-<cand>.jsonl \
      --champ-ahead result/RWKV-<champ>.jsonl --champ-imm result/RWKV-P-<champ>.jsonl

Prints per-mode stats + the gate verdict, and a final machine-readable JSON line.
Exit code 0 = both p-gates pass, 1 = at least one fails (improvement deltas are reported
but NOT gated here -- the 0.0003 check stays a separate, explicit gate).
"""
import argparse
import json
import math
import sys

from scipy.stats import wilcoxon

RAW_BAR = 0.0001      # Andrew 2026-08-10: raw improvement bar, no rounding step
NOISE_FLOOR = 7.5e-5  # iter 44: spread between same-capacity schedule variants

P_THRESHOLD = 1e-4


def load_logloss(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["user"]] = rec["metrics"]["LogLoss"]
    if not out:
        raise SystemExit(f"ERROR: no rows in {path}")
    return out


def compare(cand_path, champ_path, mode, intersect=False):
    cand = load_logloss(cand_path)
    champ = load_logloss(champ_path)
    if set(cand) != set(champ):
        if intersect:
            # --intersect serves two protocols:
            # (1) track-2 NaN-skip models (2026-07-15): each model may skip a different
            #     few mega-chunk users -- compare on the finite-user intersection;
            # (2) the VAL/TEST split (Andrew 2026-07-21): candidates eval ONLY the val
            #     half (users 5001-7500, n=2500) and pair against the champion's
            #     full-range jsonl -- the intersection IS the val set by design.
            common = set(cand) & set(champ)
            print(f"[{mode}] intersect: n_cand={len(cand)} n_champ={len(champ)} -> "
                  f"common={len(common)} (dropped {len(cand) - len(common)} cand-only, "
                  f"{len(champ) - len(common)} champ-only)")
            if len(common) < 2000:
                raise SystemExit(f"ERROR [{mode}]: intersection suspiciously small ({len(common)})")
            cand = {u: v for u, v in cand.items() if u in common}
            champ = {u: v for u, v in champ.items() if u in common}
        else:
            only_c = sorted(set(cand) - set(champ))[:5]
            only_h = sorted(set(champ) - set(cand))[:5]
            raise SystemExit(f"ERROR [{mode}]: user sets differ (n_cand={len(cand)} n_champ={len(champ)}; "
                             f"cand-only {only_c} champ-only {only_h}) -- not a paired comparison "
                             f"(use --intersect for track-2 NaN-skip models)")
    users = sorted(cand)
    # positive diff = candidate improved (lower logloss) on that user
    diffs = [champ[u] - cand[u] for u in users]
    if not all(math.isfinite(d) for d in diffs):
        raise SystemExit(f"ERROR [{mode}]: non-finite LogLoss present")
    n = len(diffs)
    mean_cand = sum(cand.values()) / n
    mean_champ = sum(champ.values()) / n
    delta = mean_champ - mean_cand  # positive = candidate better on the by-user mean
    # one-sided: H1 = candidate better (median diff > 0); zero diffs dropped (wilcox default)
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) < 10:
        p = 1.0  # models numerically identical -> no evidence of improvement
    else:
        p = float(wilcoxon(nonzero, alternative="greater").pvalue)
    # Mirror test, H1 = candidate WORSE. Needed by the curve-side rule's do-no-harm guard; it is
    # NOT 1-p (Wilcoxon is discrete and zero diffs are dropped), so it is computed, not derived.
    if len(nonzero) < 10:
        p_worse = 1.0
    else:
        p_worse = float(wilcoxon(nonzero, alternative="less").pvalue)
    return {"mode": mode, "n": n, "champ_mean": mean_champ, "cand_mean": mean_cand,
            "delta": delta, "wilcoxon_p": p, "p_pass": bool(p < P_THRESHOLD),
            "p_worse": p_worse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ahead", required=True)
    ap.add_argument("--cand-imm", required=True)
    ap.add_argument("--champ-ahead", required=True)
    ap.add_argument("--champ-imm", required=True)
    ap.add_argument("--intersect", action="store_true",
                    help="compare on the intersection of user sets (track-2: NaN-skip models)")
    ap.add_argument("--curve-side", action="store_true",
                    help="CURVE-SIDE rule (Andrew 2026-08-12): 'ahead better, imm not "
                         "statistically significantly worse'. For levers that only touch the "
                         "curve/ahead objective -- self-distillation, PAVA lambda, ahead-target "
                         "and monotonicity changes -- where the imm head's own objective "
                         "(cross-entropy on label_rating) is untouched and imm can move only "
                         "through the shared trunk. Do NOT use it for trunk/optimizer/capacity "
                         "changes, which genuinely can move both modes.")
    ap.add_argument("--harm-alpha", type=float, default=0.05,
                    help="significance level for 'imm significantly worse' (default 0.05)")
    args = ap.parse_args()

    results = [compare(args.cand_ahead, args.champ_ahead, "ahead", args.intersect),
               compare(args.cand_imm, args.champ_imm, "imm", args.intersect)]
    for r in results:
        print(f"[{r['mode']:5s}] n={r['n']}  champ={r['champ_mean']:.6f}  cand={r['cand_mean']:.6f}  "
              f"delta={r['delta']:+.6f}  wilcoxon_p={r['wilcoxon_p']:.3e}  "
              f"p<{P_THRESHOLD:g}: {'PASS' if r['p_pass'] else 'FAIL'}")
    if args.curve_side:
        ahead, imm = results[0], results[1]
        ahead_ok = ahead["delta"] >= RAW_BAR and ahead["p_pass"]
        # "imm significantly worse" = worse on the GATE METRIC (the by-user mean) AND the per-user
        # ranks agree it is worse. Both halves are load-bearing, and iter 44 is why: its imm mean
        # moved -0.000001 (nominally worse) while the RANK test said candidate BETTER at p=1e-4 --
        # most users improved slightly, a few worsened a lot. A rank-only guard would have failed a
        # magnitude-null iteration; a mean-only guard fires on noise. Requiring both means the
        # guard trips only when the decline is real in the metric we actually gate on.
        imm_worse = imm["delta"] < 0 and imm["p_worse"] < args.harm_alpha
        print(f"[curve-side] ahead delta {ahead['delta']:+.6f} >= {RAW_BAR} and p<{P_THRESHOLD:g}: "
              f"{'PASS' if ahead_ok else 'FAIL'}")
        print(f"[curve-side] imm delta {imm['delta']:+.6f}, p(worse)={imm['p_worse']:.3e} vs "
              f"alpha={args.harm_alpha:g} -> significantly worse: {'YES' if imm_worse else 'NO'}")
        if imm["delta"] < -NOISE_FLOOR and not imm_worse:
            print(f"  NOTE: imm mean fell by more than the {NOISE_FLOOR:g} same-capacity noise "
                  f"floor while the rank test did not call it worse -- a few users moved a lot. "
                  f"Passes the rule, but look before promoting.")
        ok = ahead_ok and not imm_worse
        print(f"CURVE-SIDE GATE (ahead better, imm not significantly worse): "
              f"{'PASS' if ok else 'FAIL'}")
        print("PAIRED_P_JSON " + json.dumps({r["mode"]: {"delta": round(r["delta"], 6),
                                                         "p": r["wilcoxon_p"],
                                                         "p_worse": r["p_worse"]}
                                             for r in results}))
        sys.exit(0 if ok else 1)
    all_pass = all(r["p_pass"] for r in results)
    print(f"P-GATE (both modes p<{P_THRESHOLD:g}): {'PASS' if all_pass else 'FAIL'}")
    print("PAIRED_P_JSON " + json.dumps({r["mode"]: {"delta": round(r["delta"], 6),
                                                     "p": r["wilcoxon_p"]} for r in results}))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
