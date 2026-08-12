"""The QAT-tax decomposition (Andrew's three cells, 2026-08-12).

    cell 1  plain      no QAT, full precision      -- the champion's own number
    cell 2  qat-q      QAT-trained, eval QUANTIZED -- the deploy number
    cell 3  qat-fp     QAT-trained, eval FULL PREC -- same checkpoint, QAT env off

    FULL TAX               = cell2 - cell1
    PRECISION DEGRADATION  = cell2 - cell3   what quantization costs a model trained for it
    MODEL DRIFT            = cell3 - cell1   what training under fake-quant costs by itself

Optionally a PTQ cell (plain checkpoint evaluated quantized) gives the fourth quantity:

    PTQ COST               = ptq   - cell1   quantizing a model NEVER trained for it
    RECOVERED BY QAT       = PTQ COST - PRECISION DEGRADATION

WHY THE SPLIT IS THE POINT. If the tax is mostly precision degradation, a better codebook
buys it back and the lever is quantizer capacity. If it is mostly model drift, the codebook
is irrelevant and the lever is how we TRAIN under fake-quant (placement, schedule, LR,
warm-start length). The d=32 record says drift dominated -- decay-QAT #39 had degradation
-0.000127/+0.000018 against drift +0.001129/+0.002456 -- so this report exists to check
whether that still holds at d=80 before any reduction work is aimed anywhere.

All comparisons are PAIRED on the per-user intersection, so cells evaluated on different
user ranges (e.g. a 500-user PTQ probe against a 2500-user champion eval) are still
comparable; n is reported per pair and a mismatch is never silent.

Usage:
  python optimization/qat_tax_report.py --plain iter45_kddecay --qat-q qtax_qatq \
      --qat-fp qtax_qatfp [--ptq qtax_m2b12_ptq] [--json out.json]

Tags are result-jsonl tags: result/RWKV-<tag>.jsonl (ahead) + result/RWKV-P-<tag>.jsonl (imm).
"""
import argparse
import json
import os
import sys

from scipy.stats import wilcoxon

MODES = (("ahead", "RWKV-{}.jsonl"), ("imm", "RWKV-P-{}.jsonl"))


def load(tag, pattern, resdir):
    path = os.path.join(resdir, pattern.format(tag))
    if not os.path.exists(path):
        return None, path
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["user"]] = rec["metrics"]["LogLoss"]
    return (out or None), path


def paired(a, b):
    """mean(a) - mean(b) on the common users, plus a two-sided Wilcoxon on the diffs.

    Two-sided on purpose: a tax component has no pre-registered direction (drift can be
    negative -- fake-quant is also a regularizer), so a one-sided test would beg the question.
    """
    common = sorted(set(a) & set(b))
    if not common:
        return None
    da = [a[u] for u in common]
    db = [b[u] for u in common]
    diffs = [x - y for x, y in zip(da, db)]
    delta = sum(diffs) / len(diffs)
    nonzero = [d for d in diffs if d != 0.0]
    if nonzero:
        p = wilcoxon(nonzero, alternative="two-sided").pvalue
    else:
        p = 1.0
    worse = sum(1 for d in diffs if d > 0)
    return {"n": len(common), "delta": delta, "p": p,
            "frac_worse": worse / len(common),
            "mean_a": sum(da) / len(da), "mean_b": sum(db) / len(db)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="cell 1 tag (no QAT, full precision)")
    ap.add_argument("--qat-q", required=True, help="cell 2 tag (QAT-trained, eval quantized)")
    ap.add_argument("--qat-fp", help="cell 3 tag (QAT-trained, eval full precision)")
    ap.add_argument("--ptq", help="optional: plain checkpoint evaluated quantized")
    ap.add_argument("--resdir", default="result")
    ap.add_argument("--json", help="write the machine-readable summary here too")
    args = ap.parse_args()

    report = {"cells": {}, "components": {}}
    missing = []
    for mode, pattern in MODES:
        cells = {}
        for key, tag in (("plain", args.plain), ("qat_q", args.qat_q),
                         ("qat_fp", args.qat_fp), ("ptq", args.ptq)):
            if not tag:
                continue
            data, path = load(tag, pattern, args.resdir)
            if data is None:
                missing.append(path)
            else:
                cells[key] = data

        print(f"\n===== {mode.upper()} =====")
        for key in ("plain", "qat_q", "qat_fp", "ptq"):
            if key in cells:
                d = cells[key]
                print(f"  {key:8s} n={len(d):5d}  mean {sum(d.values())/len(d):.6f}")

        comps = []
        if "qat_q" in cells and "plain" in cells:
            comps.append(("FULL TAX", "qat_q", "plain",
                          "deploy cost vs the champion as it stands"))
        if "qat_q" in cells and "qat_fp" in cells:
            comps.append(("PRECISION DEGRADATION", "qat_q", "qat_fp",
                          "what quantization costs a model trained for it"))
        if "qat_fp" in cells and "plain" in cells:
            comps.append(("MODEL DRIFT", "qat_fp", "plain",
                          "what training under fake-quant costs by itself"))
        if "ptq" in cells and "plain" in cells:
            comps.append(("PTQ COST", "ptq", "plain",
                          "quantizing a model never trained for it"))

        for label, ka, kb, blurb in comps:
            r = paired(cells[ka], cells[kb])
            if r is None:
                print(f"  {label:22s}  no common users")
                continue
            print(f"  {label:22s} {r['delta']:+.6f}   n={r['n']:5d}  p={r['p']:.3g}  "
                  f"worse on {r['frac_worse']*100:.0f}% of users")
            print(f"  {'':22s} ({blurb})")
            report["components"].setdefault(mode, {})[label] = r

        # The decomposition must close: full tax == degradation + drift (same user set).
        c = report["components"].get(mode, {})
        if all(k in c for k in ("FULL TAX", "PRECISION DEGRADATION", "MODEL DRIFT")):
            lhs = c["FULL TAX"]["delta"]
            rhs = c["PRECISION DEGRADATION"]["delta"] + c["MODEL DRIFT"]["delta"]
            # Exact only when all three cells share a user set; report the residual either way
            # so a silent range mismatch shows up as a number instead of a wrong conclusion.
            print(f"  {'CHECK':22s} tax {lhs:+.6f} vs deg+drift {rhs:+.6f} "
                  f"(residual {lhs - rhs:+.2e}{'' if abs(lhs - rhs) < 1e-9 else '  <-- user sets differ'})")
        if "PTQ COST" in c and "PRECISION DEGRADATION" in c:
            rec = c["PTQ COST"]["delta"] - c["PRECISION DEGRADATION"]["delta"]
            print(f"  {'RECOVERED BY QAT':22s} {rec:+.6f}   "
                  f"(PTQ cost minus what remains after the fine-tune)")
            c["RECOVERED BY QAT"] = {"delta": rec}

        report["cells"][mode] = {k: {"n": len(v), "mean": sum(v.values()) / len(v)}
                                 for k, v in cells.items()}

    if missing:
        print("\nMISSING (cells skipped):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
