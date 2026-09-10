"""Report cmixgraft against scratchpad/cmixgraft/PREREG.md, P3 FIRST (engagement before the number).

Control = arm 1 (w10plain) exactly: same WS checkpoint, same 21,870-step decay, same env but for
RWKV_STRIP_CMIX. Deltas are arm1 - cmixgraft, i.e. POSITIVE = the capacity helped.

Usage: python scratchpad/cmixgraft/verdict_cmixgraft.py [tag]      (default cmixgraft)
"""
import json
import math
import subprocess
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "cmixgraft"
CTRL = "w10plain"
PRE = {"ahead": "RWKV-", "imm": "RWKV-P-"}


def load(p):
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["user"]] = r
    return out


def main():
    print("P3 engagement (measured before the logloss is read)")
    ckpt = "scratchpad/ws10/w10g_d_21870.pth" if TAG == "cmixgraft" else None
    if ckpt:
        r = subprocess.run([sys.executable, "scratchpad/cmixgraft/engage_probe.py", ckpt],
                           capture_output=True, text=True)
        print("  " + "\n  ".join((r.stdout or r.stderr).strip().splitlines()))
        engaged = r.returncode == 0
    else:
        engaged = None
        print("  (non-default tag: run engage_probe.py on its checkpoint by hand)")

    d = {}
    for m in ("ahead", "imm"):
        c, b = load(f"result/{PRE[m]}{TAG}.jsonl"), load(f"result/{PRE[m]}{CTRL}.jsonl")
        us = sorted(u for u in set(c) & set(b) if math.isfinite(c[u]["metrics"]["LogLoss"]))
        size_bad = sum(c[u]["size"] != b[u]["size"] for u in us)
        mc = sum(c[u]["metrics"]["LogLoss"] for u in us) / len(us)
        mb = sum(b[u]["metrics"]["LogLoss"] for u in us) / len(us)
        d[m] = mb - mc
        print(f"\n{m:<6} n={len(us)}  size mismatches {size_bad}  arm 1 {mb:.6f}  {TAG} {mc:.6f}"
              f"  gain {d[m]:+.6f}")

    print("\nP1 capacity binds at the real budget: ahead gain >= +0.0003")
    print(f"  ahead {d['ahead']:+.6f} -> {'HOLDS' if d['ahead'] >= 0.0003 else 'does not hold'}")
    print("P2 imm moves less than ahead")
    print(f"  |imm| {abs(d['imm']):.6f} vs |ahead| {abs(d['ahead']):.6f} -> "
          f"{'HOLDS' if abs(d['imm']) < abs(d['ahead']) else 'does not hold'}")
    print("P4 falsifier: ahead within +/-0.0001 WITH P3 satisfied")
    if engaged is False:
        print("  P3 FAILED -> uninterpretable; do NOT read the gains above as a null")
    elif abs(d["ahead"]) <= 0.0001:
        print("  FIRES: two epochs of coarse capacity buy nothing on ahead. Remember the asymmetry --")
        print("  a null here is WEAK (2 epochs for brand-new parameters); it is not 'capacity does not bind'")
    else:
        print("  does not fire")

    cmd = [sys.executable, "optimization/paired_pvalue.py",
           "--cand-ahead", f"result/RWKV-{TAG}.jsonl", "--cand-imm", f"result/RWKV-P-{TAG}.jsonl",
           "--champ-ahead", f"result/RWKV-{CTRL}.jsonl", "--champ-imm", f"result/RWKV-P-{CTRL}.jsonl",
           "--intersect"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"\nboth-modes gate vs arm 1 (paired_pvalue exit {r.returncode}; informational -- an"
              " endgame branch, not a champion candidate):")
        print((r.stdout or r.stderr).strip()[:1500])
    except Exception as exc:
        print(f"\npaired_pvalue failed: {exc}")


if __name__ == "__main__":
    main()
