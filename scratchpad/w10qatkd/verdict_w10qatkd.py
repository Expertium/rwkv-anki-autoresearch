"""Report w10qatkd against scratchpad/w10qatkd/PREREG.md. Both arms are quant-aware.

  gain      = arm 2 (w10qat) - w10qatkd       POSITIVE = KD helped
  tax left  = w10qatkd - arm 1 (w10plain)     what quantization still costs with KD
Usage: python scratchpad/w10qatkd/verdict_w10qatkd.py
"""
import json
import math
import subprocess
import sys

PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}


def load(p):
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["user"]] = r
    return out


def mean_on(d, us):
    return sum(d[u]["metrics"]["LogLoss"] for u in us) / len(us)


def main():
    g = {}
    for m in ("ahead", "imm"):
        k, q, p = (load(f"result/{PRE[m]}{t}.jsonl") for t in ("w10qatkd", "w10qat", "w10plain"))
        us = sorted(u for u in set(k) & set(q) & set(p) if math.isfinite(k[u]["metrics"]["LogLoss"]))
        mk, mq, mp = mean_on(k, us), mean_on(q, us), mean_on(p, us)
        size_bad = sum(k[u]["size"] != q[u]["size"] for u in us)
        g[m] = mq - mk
        tax2, taxk = mq - mp, mk - mp
        frac = (tax2 - taxk) / tax2 if tax2 else float("nan")
        print(f"{m:<6} n={len(us)} size!={size_bad}  arm1 {mp:.6f}  arm2 {mq:.6f}  kd {mk:.6f}")
        print(f"       KD gain {g[m]:+.6f}   tax: arm 2 {tax2:+.6f} -> with KD {taxk:+.6f}"
              f"   ({frac:.0%} of the tax recovered)")
    print("\nP1 KD shrinks the tax in BOTH modes by >= +0.0001 (p-gate below)")
    print(f"  ahead {g['ahead']:+.6f}  imm {g['imm']:+.6f} -> "
          f"{'magnitudes clear' if min(g.values()) >= 0.0001 else 'magnitudes do NOT clear'}")
    print(f"P2 imm recovers more: {'yes' if g['imm'] > g['ahead'] else 'no'}")
    if max(abs(x) for x in g.values()) <= 0.0001:
        print("P4 FIRES: both within +/-0.0001 -- KD from the fp twin does not shrink the tax here")
    cmd = [sys.executable, "optimization/paired_pvalue.py",
           "--cand-ahead", "result/RWKV-w10qatkd.jsonl", "--cand-imm", "result/RWKV-P-w10qatkd.jsonl",
           "--champ-ahead", "result/RWKV-w10qat.jsonl", "--champ-imm", "result/RWKV-P-w10qat.jsonl",
           "--intersect"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(f"\npaired Wilcoxon vs arm 2 (exit {r.returncode}):")
    print((r.stdout or r.stderr).strip()[:1500])


if __name__ == "__main__":
    main()
