"""Report arm 1 against everything PREREG.md commits to, including the unflattering lines.

Usage: python scratchpad/w10plain/verdict_w10plain.py [tag]     (default w10plain)
"""
import json
import subprocess
import sys
from pathlib import Path

TAG = sys.argv[1] if len(sys.argv) > 1 else "w10plain"
REF = {"realcyc": (0.298083, 0.263592), "base5k(published basis)": (0.294623, 0.263586)}
CRIT = (0.2950, 0.2640)


def load(p):
    out = {}
    for ln in open(p):
        r = json.loads(ln)
        out[r["user"]] = r
    return out


def main():
    a = load(f"result/RWKV-{TAG}.jsonl")
    i = load(f"result/RWKV-P-{TAG}.jsonl")
    users = sorted(set(a) & set(i))
    ma = sum(a[u]["metrics"]["LogLoss"] for u in users) / len(users)
    mi = sum(i[u]["metrics"]["LogLoss"] for u in users) / len(users)
    nan = sum(1 for u in users if a[u]["metrics"]["LogLoss"] != a[u]["metrics"]["LogLoss"])
    print(f"{TAG}: n={len(users)}  ahead {ma:.6f}  imm {mi:.6f}  nan_users {nan}\n")

    # size is a DATA-INTEGRITY check, never a result: any mismatch is a pipeline bug.
    base = Path("optimization/size_baseline_id_e2s.json")
    if base.exists():
        snap = json.loads(base.read_text())
        sizes = snap.get("sizes", snap)
        bad = [u for u in users if str(u) in sizes and a[u]["size"] != sizes[str(u)]]
        print(f"size vs the id_e2s baseline: {len(bad)}/{len(users)} mismatches"
              f"{'  <-- PIPELINE BUG' if bad else ''}")
    else:
        print(f"size baseline {base} missing -- snapshot it before gating anything on size")

    print(f"\n{'reference':<26}{'d ahead':>10}{'d imm':>10}   (positive = arm 1 is better)")
    for name, (ra, ri) in REF.items():
        print(f"{name:<26}{ra - ma:>+10.6f}{ri - mi:>+10.6f}")

    d = REF["realcyc"][0] - ma
    print("\nQ1 budget premise (CLEAN: same model, same basis, 1+1 -> 10+2 epochs)")
    print(f"  measured ahead gain {d:+.6f} against a projected +0.0042")
    if d >= 0.0035:
        print("  => the premise HOLDS; phase 6 earned its ~4 days")
    elif d >= 0.0015:
        print("  => real but well under projection: the log-linear extrapolation overstated 10x")
    else:
        print("  => REFUTED at this budget. Architecture, out-of-family structure and budget are")
        print("     then all exhausted, and the stop criterion is unreachable by any known lever.")
        print("     That is a fork for Andrew, not a result to work around.")

    print("\nQ2 capacity at the real budget (NOT clean: the d=128 basis differs by ~0.0001)")
    if ma <= 0.2950:
        print(f"  ahead {ma:.6f} <= 0.2950: 563k matches 2.76M at a matched budget; capacity does")
        print("  not bind and fixing the model size cost nothing")
    elif ma >= 0.2965:
        print(f"  ahead {ma:.6f} >= 0.2965: the ~0.003 gap survives 10x the budget, so capacity is")
        print("  a live suspect for the first time -- Andrew's call, he fixed the size for this run")
    else:
        print(f"  ahead {ma:.6f} lands between the pre-registered lines: no verdict, say so")

    print("\nQ3 stop criterion (its basis PREDATES gen 5 -- flagged, not silently applied)")
    print(f"  ahead {ma:.6f} vs <= {CRIT[0]}   {'MET' if ma <= CRIT[0] else 'NOT met'}")
    print(f"  imm   {mi:.6f} vs <= {CRIT[1]}   {'MET' if mi <= CRIT[1] else 'NOT met'}")

    cmd = [sys.executable, "optimization/paired_pvalue.py",
           "--cand-ahead", "result/RWKV-%s.jsonl" % TAG,
           "--cand-imm", "result/RWKV-P-%s.jsonl" % TAG,
           "--champ-ahead", "result/RWKV-realcyc.jsonl",
           "--champ-imm", "result/RWKV-P-realcyc.jsonl", "--intersect"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print("")
        print("paired Wilcoxon vs realcyc (exit %d):" % r.returncode)
        print((r.stdout or r.stderr).strip()[:1200])
    except Exception as exc:  # a verdict must not die on a tool signature change
        print("")
        print("paired_pvalue failed: %s" % exc)


if __name__ == "__main__":
    main()
