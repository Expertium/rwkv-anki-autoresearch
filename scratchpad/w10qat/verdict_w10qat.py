"""Report arm 2 against scratchpad/w10qat/PREREG.md: the QAT tax at 10+2 epochs, all four questions.

The tax is arm 2 MINUS arm 1 (positive = quantization costs that much). Both runs share the WS
checkpoint, the decay length and every env var but the eight RWKV_QAT_* ones, so this difference
needs no correction. The 1.25-epoch tax is printed beside it as a PRIOR, never as a control.

Usage: python scratchpad/w10qat/verdict_w10qat.py [tag]     (default w10qat)
"""
import json
import math
import sys
from pathlib import Path

from scipy.stats import wilcoxon

TAG = sys.argv[1] if len(sys.argv) > 1 else "w10qat"
CTRL = "w10plain"
PRIOR = {"ahead": 0.002286, "imm": 0.003486}     # qtaxd_cblearn - iter45_kddecay, 1+1 ep
CRIT = {"ahead": 0.2950, "imm": 0.2640}
BIG = {"ahead": 0.294623, "imm": 0.263586}       # old d=128, FULL PRECISION, published basis
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}


def load(p):
    out = {}
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            out[r["user"]] = r
    return out


def main():
    res = {}
    for mode in ("ahead", "imm"):
        c = load(f"result/{PRE[mode]}{TAG}.jsonl")
        b = load(f"result/{PRE[mode]}{CTRL}.jsonl")
        users = sorted(set(c) & set(b))
        nan = [u for u in users if not math.isfinite(c[u]["metrics"]["LogLoss"])]
        users = [u for u in users if u not in nan and math.isfinite(b[u]["metrics"]["LogLoss"])]
        size_bad = [u for u in users if c[u]["size"] != b[u]["size"]]
        d = [c[u]["metrics"]["LogLoss"] - b[u]["metrics"]["LogLoss"] for u in users]
        mc = sum(c[u]["metrics"]["LogLoss"] for u in users) / len(users)
        mb = sum(b[u]["metrics"]["LogLoss"] for u in users) / len(users)
        p_worse = wilcoxon(d, alternative="greater").pvalue
        res[mode] = dict(n=len(users), nan=len(nan), size_bad=len(size_bad), deploy=mc, plain=mb,
                         tax=mc - mb, p_worse=p_worse, frac_worse=sum(x > 0 for x in d) / len(d))

    print(f"{TAG} vs {CTRL}   (tax = arm 2 - arm 1; positive = quantization costs that much)\n")
    print(f"{'mode':<7}{'n':>6}{'nan':>5}{'size!=':>8}{'arm 1':>11}{'arm 2':>11}{'TAX':>11}"
          f"{'1.25-ep prior':>15}{'users worse':>13}{'p(worse)':>11}")
    for m, r in res.items():
        print(f"{m:<7}{r['n']:>6}{r['nan']:>5}{r['size_bad']:>8}{r['plain']:>11.6f}{r['deploy']:>11.6f}"
              f"{r['tax']:>+11.6f}{PRIOR[m]:>+15.6f}{r['frac_worse']:>12.1%}{r['p_worse']:>11.2e}")
    if any(r["size_bad"] for r in res.values()):
        print("\n!! size differs user-for-user between the arms -- a PIPELINE BUG, not a result")

    base = Path("optimization/size_baseline_id_e2s.json")
    if base.exists():
        snap = json.loads(base.read_text())
        sizes = snap.get("sizes", snap)
        c = load(f"result/RWKV-{TAG}.jsonl")
        bad = [u for u in c if str(u) in sizes and c[u]["size"] != sizes[str(u)]]
        print(f"size vs the id_e2s baseline: {len(bad)}/{len(c)} mismatches"
              f"{'  <-- PIPELINE BUG' if bad else ''}")

    ta = res["ahead"]["tax"]
    print("\nQ1 does the 10x budget change the QAT tax?  (pre-registered on AHEAD)")
    if ta < 0.0015:
        print(f"  ahead tax {ta:+.6f} < +0.0015: the budget SHRINKS the tax")
    elif ta <= 0.0030:
        print(f"  ahead tax {ta:+.6f} in [+0.0015, +0.0030]: unchanged within tolerance; the 1.25-ep")
        print("  tax transfers and the retry attacks the same terms")
    else:
        print(f"  ahead tax {ta:+.6f} > +0.0030: the budget GROWS the tax -- the largest lever left,")
        print("  and Andrew's conditional byte-budget grant comes forward")
    ti = res["imm"]["tax"]
    print(f"  (imm tax {ti:+.6f}; its pre-registered band was +0.0024..+0.0045)")

    print("\nQ2 which mode pays more?  prediction: imm")
    print(f"  imm {ti:+.6f} vs ahead {ta:+.6f} -> {'imm pays more, as predicted' if ti > ta else 'AHEAD pays more -- new'}")

    print("\nQ3 the DEPLOYED numbers against the stop criterion (its basis predates gen 5 -- flagged)")
    for m, r in res.items():
        ok = r["deploy"] <= CRIT[m]
        print(f"  {m:<6}{r['deploy']:.6f} vs <= {CRIT[m]}   {'MET' if ok else 'NOT met'}"
              f"   (margin {CRIT[m] - r['deploy']:+.6f})")
    print("  context only, NOT a fair comparison -- the old d=128 model is FULL PRECISION on the")
    print("  published basis:")
    for m, r in res.items():
        print(f"    {m:<6}deploy {r['deploy']:.6f} vs old fp32 {BIG[m]:.6f}  ({BIG[m] - r['deploy']:+.6f})")

    print("\nQ4 engagement -- the probe (users 5001-5010), checked by the runner before the full eval")
    try:
        for m in ("ahead", "imm"):
            pc = load(f"result/{PRE[m]}{TAG}p.jsonl")
            b = load(f"result/{PRE[m]}{CTRL}.jsonl")
            us = sorted(set(pc) & set(b))
            cost = sum(pc[u]["metrics"]["LogLoss"] - b[u]["metrics"]["LogLoss"] for u in us) / len(us)
            print(f"  {m:<6}probe cost {cost:+.6f} on n={len(us)}")
    except OSError as exc:
        print(f"  probe results missing: {exc}")
    print("  (the loaded catalog PATHS are in scratchpad/w10qat/probe_*.log -- the runner refuses")
    print("   to start the full eval unless they are the step-21870 learned ones)")


if __name__ == "__main__":
    main()
