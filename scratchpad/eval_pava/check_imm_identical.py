"""imm must be BIT-IDENTICAL between a rectified and an unrectified eval of the same model.

Why this is a real test and not a tautology: the rectified eval inserts four counterfactual
probe rows before every scored review. Those are skip rows, and the WKV kernel restores the
pre-step state on a skip (rwkv7_cuda.cu: `if (skip) state_xy = in_state_xy`), so they must
perturb nothing. imm comes from the RATING head, which the rectifier never touches -- so any
imm difference at all means the probes ARE perturbing the stream, which would silently
invalidate the ahead numbers too.

This checks that claim on the full val half rather than on my reading of the kernel.

Usage: python scratchpad/eval_pava/check_imm_identical.py <plain.jsonl> <rect.jsonl>
"""
import json
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["user"]] = r
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if not common:
        print("IMM_CHECK_FAILED: no users in common")
        sys.exit(1)
    only = (set(a) ^ set(b))
    worst_u, worst = None, 0.0
    nsize = 0
    for u in common:
        d = abs(a[u]["metrics"]["LogLoss"] - b[u]["metrics"]["LogLoss"])
        if d > worst:
            worst, worst_u = d, u
        if a[u].get("size") != b[u].get("size"):
            nsize += 1
    print(f"users compared: {len(common)}   only-in-one: {len(only)}")
    print(f"size mismatches: {nsize}")
    print(f"worst |dLogLoss|: {worst:.3e}" + (f" (user {worst_u})" if worst_u else ""))
    ok = worst == 0.0 and nsize == 0 and not only
    if not ok and worst > 0.0:
        print("  ^ NON-ZERO: the probes are perturbing the stream. The ahead numbers from "
              "the rectified eval are then NOT trustworthy either -- stop and diagnose.")
    print("IMM_CHECK_" + ("IDENTICAL" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
