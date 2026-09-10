"""Report the leak-free branch against everything PREREG.md commits to, unflattering lines included.

Usage: python scratchpad/w10clk/verdict_w10clk.py [tag]     (default w10clk)
"""
import json
import subprocess
import sys
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "w10clk"
CRIT = (0.2950, 0.2640)
PL_USERS = range(5001, 5301)          # phase L's users


def load(p):
    out = {}
    for ln in open(p):
        r = json.loads(ln)
        out[r["user"]] = r
    return out


def mean(d, users):
    return sum(d[u]["metrics"]["LogLoss"] for u in users) / len(users)


def main():
    a, i = load(f"result/RWKV-{TAG}.jsonl"), load(f"result/RWKV-P-{TAG}.jsonl")
    a1, i1 = load("result/RWKV-w10plain.jsonl"), load("result/RWKV-P-w10plain.jsonl")
    users = sorted(set(a) & set(i) & set(a1) & set(i1))
    ma, mi, ma1, mi1 = mean(a, users), mean(i, users), mean(a1, users), mean(i1, users)
    nan = sum(1 for u in users if a[u]["metrics"]["LogLoss"] != a[u]["metrics"]["LogLoss"])
    print(f"{TAG}: n={len(users)}  ahead {ma:.6f}  imm {mi:.6f}  nan_users {nan}")
    sz = [u for u in users if a[u]["size"] != a1[u]["size"] or i[u]["size"] != i1[u]["size"]]
    print(f"size vs arm 1: {len(sz)}/{len(users)} mismatches{'  <-- PIPELINE BUG' if sz else ''}")
    if Path(f"scratchpad/{TAG}/EVAL_BANNER_MISSING.txt").exists():
        print("⚠ EVAL_BANNER_MISSING.txt exists: the eval may have run WITHOUT the fix -- stop here")

    print(f"\nP1 the leak's value after adaptation (positive = the leak-free branch is WORSE)")
    print(f"  ahead {ma - ma1:+.6f}  (band +0.0001..+0.0008)")
    print(f"  imm   {mi - mi1:+.6f}  (band +0.0010..+0.0040)")

    print("\nP2 ordered against phase L on imm, users 5001-5300")
    try:
        c0 = load("result/RWKV-P-lkc0.jsonl")
        c30 = load("result/RWKV-P-lkc30m.jsonl")
        pu = sorted(set(PL_USERS) & set(i) & set(i1) & set(c0) & set(c30))
        br = mean(i, pu) - mean(i1, pu)
        pl = mean(c30, pu) - mean(c0, pu)
        print(f"  n={len(pu)}  branch {br:+.6f}   lkc30m {pl:+.6f}   ratio {br / pl if pl else float('nan'):.2f}"
              f"  (expect <= 1, likely 0.4-0.8)")
        if pl and br > pl:
            print("  ⚠ the leak-free decay did WORSE than no adaptation: a defect to find, not a result")
    except FileNotFoundError as e:
        print(f"  phase L results missing ({e.filename}) -- cannot check")

    print("\nP3 stop criterion")
    print(f"  ahead {ma:.6f} vs <= {CRIT[0]}   {'MET' if ma <= CRIT[0] else 'NOT met'}")
    print(f"  imm   {mi:.6f} vs <= {CRIT[1]}   {'MET' if mi <= CRIT[1] else 'NOT met'}"
          f"   (arm 1 with the leak: {mi1:.6f})")

    print("\nP4 the old d=128 model (no timestamp features, so no leak), same users")
    try:
        ba, bi = load("result/RWKV-base5k.jsonl"), load("result/RWKV-P-base5k.jsonl")
        bu = sorted(set(users) & set(ba) & set(bi))
        print(f"  n={len(bu)}  d ahead {mean(ba, bu) - mean(a, bu):+.6f}  d imm {mean(bi, bu) - mean(i, bu):+.6f}"
              f"   (positive = w10clk beats d=128)")
    except FileNotFoundError as e:
        print(f"  {e.filename} missing")

    print("\nP5 engagement (from the decay log)")
    for p in sorted(Path(f"scratchpad/{TAG}").glob("decay_*.log")):
        lines = [l for l in p.read_text(errors="replace").splitlines() if "[clock-fix pid" in l]
        print(f"  {p.name}: {len(lines)} clock-fix lines; last: {lines[-1][:160] if lines else 'NONE'}")

    cmd = [sys.executable, "optimization/paired_pvalue.py",
           "--cand-ahead", f"result/RWKV-{TAG}.jsonl", "--cand-imm", f"result/RWKV-P-{TAG}.jsonl",
           "--champ-ahead", "result/RWKV-w10plain.jsonl", "--champ-imm", "result/RWKV-P-w10plain.jsonl",
           "--intersect"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print("\npaired Wilcoxon vs arm 1 (candidate better = the leak was NOT worth anything):")
    print(r.stdout[-1500:])


if __name__ == "__main__":
    main()
