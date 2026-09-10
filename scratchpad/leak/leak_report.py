"""Report the query-row clock-leak counterfactual against scratchpad/leak/PREREG.md.

Arms (arm 1's checkpoint, users 5001-5300, rectified metric, same wrapper path):
  lkc0    LEAK_CF_T=0      control -- the wrapper with the transform OFF
  lkc30m  LEAK_CF_T=1800   in-session gaps (< 30 min) moved to "card shown at the previous answer"
  lkc3h   LEAK_CF_T=10800  the same below 3 h
positive = the counterfactual (deploy-faithful) LogLoss is HIGHER, i.e. the eval number overstates
what this model does at deploy by that much.
Usage: python scratchpad/leak/leak_report.py
"""
import json
import math
import os
import sys

from scipy.stats import wilcoxon

RES = os.environ.get("LEAK_RESULT_DIR", "result")
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}
LO, HI = 5001, 5300


def load(p):
    if not os.path.exists(p):
        return None
    return {r["user"]: r["metrics"]["LogLoss"] for r in
            (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}


def main():
    ok = True
    print("QUERY-ROW CLOCK LEAK -- counterfactual minus control, arm 1 checkpoint, users 5001-5300\n")
    for mode in ("ahead", "imm"):
        ctrl = load(f"{RES}/{PRE[mode]}lkc0.jsonl")
        full = load(f"{RES}/{PRE[mode]}w10plain.jsonl")
        if ctrl is None:
            print(f"{mode}: control missing"); ok = False; continue
        if full:
            com = [u for u in ctrl if u in full and math.isfinite(ctrl[u]) and math.isfinite(full[u])]
            mx = max(abs(ctrl[u] - full[u]) for u in com) if com else float("nan")
            print(f"{mode:5s} harness: control vs arm 1's own full run, {len(com)} users, max |d| {mx:.2e}")
        for arm in ("lkc30m", "lkc3h"):
            a = load(f"{RES}/{PRE[mode]}{arm}.jsonl")
            if a is None:
                print(f"{mode:5s} {arm}: missing"); continue
            us = [u for u in range(LO, HI + 1) if u in a and u in ctrl
                  and math.isfinite(a[u]) and math.isfinite(ctrl[u])]
            d = [a[u] - ctrl[u] for u in us]
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
            nz = [x for x in d if x != 0]
            p = wilcoxon(nz).pvalue if len(nz) >= 10 else float("nan")
            up = sum(x > 0 for x in d)
            print(f"{mode:5s} {arm:7s} {m:+.6f} +/- {se:.6f}  (n={len(us)}, worse on {up}, p={p:.1e})")
    print("\nPREREG decision (imm at 30 min): >= +0.0010 material | +0.0003..+0.0010 secondary | < +0.0003 minor")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
