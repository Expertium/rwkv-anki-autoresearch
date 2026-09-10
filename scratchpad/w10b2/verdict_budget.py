"""The budget curve: w10b2 (2.01+2 ep), w10b5 (5.03+2), arm 1 (10+2), all from ONE WS run.

Read on users 5001-5300 only (the curve's eval set). These users are HARDER than average (+0.0035
on ahead), so only PAIRED differences mean anything here: never quote a level, and never compare a
curve point to the 0.2950 stop criterion. Paired SE on 300 users is ~0.0003 (mk_w10budget.py).
realcyc (1+1 ep, its own WS) is printed as corroboration, NOT as a curve point: different warmup
share, and a 1-epoch decay.

Usage: python scratchpad/w10b2/verdict_budget.py
"""
import json
import math

POINTS = [("w10b2", 2.01), ("w10b5", 5.03), ("w10plain", 10.0)]
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}


def load(p):
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["user"]] = r["metrics"]["LogLoss"]
    return out


def main():
    for m in ("ahead", "imm"):
        runs = {}
        for tag, _ in POINTS + [("realcyc", None)]:
            try:
                runs[tag] = load(f"result/{PRE[m]}{tag}.jsonl")
            except OSError:
                print(f"{m}: {tag} missing")
        if not all(t in runs for t, _ in POINTS):
            continue
        us = sorted(set.intersection(*(set(runs[t]) for t, _ in POINTS)))
        us = [u for u in us if 5001 <= u <= 5300 and all(math.isfinite(runs[t][u]) for t, _ in POINTS)]
        ref = runs[POINTS[-1][0]]
        print(f"\n{m}  (n={len(us)}; gain vs the 10+2 endpoint, NEGATIVE = worse than arm 1)")
        prev = None
        for tag, ws in POINTS + [("realcyc", None)]:
            if tag not in runs or not all(u in runs[tag] for u in us):
                continue
            diffs = [ref[u] - runs[tag][u] for u in us]
            mu = sum(diffs) / len(diffs)
            se = (sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs)) ** 0.5
            lab = f"WS {ws:5.2f} ep" if ws else "realcyc 1+1"
            print(f"  {tag:<9} {lab:<12} {mu:+.6f} +/- {se:.6f}")
        (t2, e2), (t5, e5), (t10, e10) = POINTS
        g2 = sum(ref[u] - runs[t2][u] for u in us) / len(us)   # = L(10) - L(2), <= 0 if budget helps
        g5 = sum(ref[u] - runs[t5][u] for u in us) / len(us)
        # LOSS DROP per doubling of WS budget on each segment (POSITIVE = loss still falls).
        # Equal under log-linear scaling; a much smaller second number means saturating.
        s1 = (g5 - g2) / math.log2(e5 / e2)     # = (L(2) - L(5)) / doublings
        s2 = (0.0 - g5) / math.log2(e10 / e5)   # = (L(5) - L(10)) / doublings
        print(f"  loss drop per doubling of WS budget: {e2:g}->{e5:g} ep {s1:+.6f}, {e5:g}->{e10:g} ep {s2:+.6f}")


if __name__ == "__main__":
    main()
