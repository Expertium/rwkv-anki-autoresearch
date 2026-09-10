"""Apply the PRE-REGISTERED leak decision mechanically, for the chain waiter.

Rules (scratchpad/leak/PREREG.md and scratchpad/w10clk/PREREG.md, both written before phase L ran):
  * imm at 30 min (lkc30m - lkc0, users 5001-5300) >= +0.0010  => MATERIAL => insert w10clk
  * and if lkc3h costs >= +0.0005 imm MORE than lkc30m          => run it at T = 10800, not 1800
Before either, the HARNESS must hold: the control (the wrapper with the transform OFF) must
reproduce arm 1's own eval on the same users within 0.0003 in both modes. If it does not, the
counterfactual measured the wrapper, not the leak, and nothing is decided automatically.

Exit codes: 0 = insert at T=1800 | 3 = insert at T=10800 | 1 = below MATERIAL, do not insert |
            2 = cannot decide (missing results, or the harness failed) -- a human looks.
Usage: python scratchpad/w10clk/clk_decide.py
"""
import json
import math
import os
import sys

RES = os.environ.get("LEAK_RESULT_DIR", "result")
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}
USERS = range(5001, 5301)
MATERIAL, SWITCH, HARNESS = 0.0010, 0.0005, 0.0003


def load(p):
    if not os.path.exists(p):
        return None
    return {r["user"]: r["metrics"]["LogLoss"] for r in
            (json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip())}


def mdiff(a, b):
    us = [u for u in USERS if u in a and u in b and math.isfinite(a[u]) and math.isfinite(b[u])]
    return (sum(a[u] - b[u] for u in us) / len(us), len(us)) if us else (float("nan"), 0)


def main():
    d = {}
    for mode, pre in PRE.items():
        for arm in ("lkc0", "lkc30m", "lkc3h", "w10plain"):
            d[mode, arm] = load(f"{RES}/{pre}{arm}.jsonl")
    missing = [f"{m}/{a}" for (m, a), v in d.items() if v is None]
    if missing:
        print(f"CANNOT DECIDE: missing {missing}")
        return 2
    for mode in PRE:
        h, n = mdiff(d[mode, "lkc0"], d[mode, "w10plain"])
        print(f"harness {mode}: control - arm 1 = {h:+.6f} (n={n})")
        if not (n >= 250 and abs(h) <= HARNESS):
            print(f"CANNOT DECIDE: the harness does not reproduce arm 1 on {mode}")
            return 2
    i30, n30 = mdiff(d["imm", "lkc30m"], d["imm", "lkc0"])
    i3h, n3h = mdiff(d["imm", "lkc3h"], d["imm", "lkc0"])
    a30, _ = mdiff(d["ahead", "lkc30m"], d["ahead", "lkc0"])
    print(f"imm   30 min {i30:+.6f} (n={n30})   3 h {i3h:+.6f} (n={n3h})   ahead 30 min {a30:+.6f}")
    if not i30 >= MATERIAL:
        print(f"BELOW MATERIAL (imm {i30:+.6f} < +{MATERIAL}): w10clk is NOT inserted ahead of the queue")
        return 1
    if i3h - i30 >= SWITCH:
        print(f"MATERIAL, and the 3 h arm costs {i3h - i30:+.6f} imm more: insert w10clk at T = 10800")
        return 3
    print(f"MATERIAL: insert w10clk at T = 1800 (3 h adds {i3h - i30:+.6f}, under +{SWITCH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
