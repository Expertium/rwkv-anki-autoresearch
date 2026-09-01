"""The track-2 RATIO gate: is a SMALLER model's accuracy cost worth the parameters it saved?

Andrew reinstated this rule for the <=100k hybrid phase (2026-08-28): *"use the same `ratio` =
`100,000 * dLL/dparams` rule as before"*. It is the gate recorded at `optimization/research_5k.md`
line 17 and the one the A0 -> A18 width ladder was judged by.

    ratio(mode) == 100,000 * (LL_cand - LL_champ) / (params_champ - params_cand)

ACCEPT iff params STRICTLY DECREASE and ratio <= 0.0001 in BOTH modes.

Read it as a PRICE: "how much LogLoss did each 100,000 parameters removed cost?" The champion's
own lineage paid 0.0000435 ahead / 0.0000240 imm across the whole 2.76M -> 558k ladder, so the
0.0001 bar is roughly twice the historical rate, not a formality.

WHY A TOOL AND NOT ARITHMETIC IN A CHAT MESSAGE. Written BEFORE the runs report, on purpose. A
gate computed after seeing the numbers is a gate with a free parameter in it, and this repo has
the scar: the accept bar drifted from >=0.0003 to a rounded 0.00005 to a raw 0.0001, each time
defensibly, and only the written-down version made the drift visible.

THREE THINGS IT DOES THAT HAND ARITHMETIC KEEPS GETTING WRONG:

  1. SIGN. A candidate that is smaller AND better has a NEGATIVE numerator, so its ratio is
     negative and passes. That is correct and not a bug -- do not take an absolute value.
  2. DIRECTION OF dparams. It must be champ - cand, i.e. params SAVED. A candidate that grew
     makes the denominator negative and flips the comparison silently, so growth is refused
     outright rather than divided by.
  3. PAIRING. Candidates eval only the VAL half (5001-7500) while the champion's jsonl may span
     the full range, so --intersect is the normal case here, not an escape hatch.

THE p-VALUES ARE REPORTED BUT DO NOT GATE. The ratio rule is a magnitude rule; the accuracy-accept
p-gate (p < 0.0001 both modes) belongs to the "candidate is BETTER" protocol, which a shrink arm
is not claiming. They are printed because a regression that is inside the +/-7.5e-5 noise floor
means something quite different from one that is real and large, and the verdict line says which.

Usage:
  python optimization/ratio_gate.py --params-cand 84007 --params-champ 558212 \
      --cand-ahead result/RWKV-hybA.jsonl --cand-imm result/RWKV-P-hybA.jsonl \
      --champ-ahead result/RWKV-iter53_muonlora.jsonl \
      --champ-imm result/RWKV-P-iter53_muonlora.jsonl --intersect

Exit 0 = ACCEPT, 1 = REJECT, 2 = usage/data error. Zero GPU cost: it reads the result jsonls.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.paired_pvalue import NOISE_FLOOR, compare  # noqa: E402

RATIO_BAR = 0.0001  # LogLoss per 100,000 parameters removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ahead", required=True)
    ap.add_argument("--cand-imm", required=True)
    ap.add_argument("--champ-ahead", required=True)
    ap.add_argument("--champ-imm", required=True)
    ap.add_argument("--params-cand", type=int, required=True)
    ap.add_argument("--params-champ", type=int, required=True)
    ap.add_argument("--intersect", action="store_true",
                    help="pair on the common users -- the normal case, since candidates eval "
                         "only the VAL half while the champion's jsonl may span the full range")
    ap.add_argument("--bar", type=float, default=RATIO_BAR)
    ap.add_argument("--label", default="candidate")
    args = ap.parse_args()

    dparams = args.params_champ - args.params_cand
    print(f"=== RATIO GATE: {args.label} ===")
    print(f"params: champion {args.params_champ:,} -> candidate {args.params_cand:,}  "
          f"(saved {dparams:,})")

    if dparams <= 0:
        # Refused, not divided by. A non-positive denominator flips the inequality's direction,
        # so a grown model would "pass" whenever it was worse -- silently, and in the right-hand
        # column of a table that says ACCEPT.
        print(f"REJECT: params did not strictly decrease ({dparams:+,}). The ratio rule prices "
              "a SHRINK; it is undefined for a model that grew.")
        print(json.dumps({"verdict": "reject", "reason": "params_not_reduced",
                          "dparams": dparams}))
        return 1

    rows, fails, notes = [], [], []
    for mode, cand, champ in (("ahead", args.cand_ahead, args.champ_ahead),
                              ("imm", args.cand_imm, args.champ_imm)):
        r = compare(cand, champ, mode, intersect=args.intersect)
        # compare() returns delta = champ_mean - cand_mean, i.e. POSITIVE when the candidate is
        # better. The ratio wants the cost, so the numerator is the negation.
        cost = -r["delta"]
        ratio = 100000.0 * cost / dparams
        ok = ratio <= args.bar
        rows.append({"mode": mode, "n": r["n"], "champ": r["champ_mean"],
                     "cand": r["cand_mean"], "cost": cost, "ratio": ratio,
                     "pass": ok, "p_better": r["wilcoxon_p"], "p_worse": r["p_worse"]})
        if not ok:
            fails.append(mode)
        if cost > 0 and abs(cost) < NOISE_FLOOR:
            notes.append(f"{mode}: the regression ({cost:+.6f}) is INSIDE the "
                         f"+/-{NOISE_FLOOR:.1e} noise floor")
        if cost < 0:
            notes.append(f"{mode}: candidate is BETTER as well as smaller ({cost:+.6f})")

    print()
    print(f"{'mode':<6} {'n':>5} {'champion':>10} {'candidate':>10} {'cost':>10} "
          f"{'ratio':>10} {'bar':>8}  verdict")
    for r in rows:
        print(f"{r['mode']:<6} {r['n']:>5} {r['champ']:>10.6f} {r['cand']:>10.6f} "
              f"{r['cost']:>+10.6f} {r['ratio']:>10.6f} {args.bar:>8.6f}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    print()
    print(f"max tolerable cost at this size: {args.bar * dparams / 100000.0:.6f} per mode")
    for note in notes:
        print("  note: " + note)
    print("  (p-values are context, not a gate: "
          + ", ".join(f"{r['mode']} p_better={r['p_better']:.2e} p_worse={r['p_worse']:.2e}"
                      for r in rows) + ")")

    verdict = "accept" if not fails else "reject"
    print()
    if fails:
        print(f"REJECT: ratio exceeds {args.bar} in {', '.join(fails)}")
    else:
        print(f"ACCEPT: ratio within {args.bar} in BOTH modes at {dparams:,} params saved")
    print(json.dumps({"verdict": verdict, "dparams": dparams, "bar": args.bar,
                      "modes": {r["mode"]: {"cost": r["cost"], "ratio": r["ratio"],
                                            "pass": r["pass"]} for r in rows}}))
    return 0 if verdict == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
