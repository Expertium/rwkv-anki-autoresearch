"""What does the d=128 KD teacher lose by not seeing `scaled_state`?

Written before the arms ran. See scratchpad/teacher_114/PLAN.md for the pre-registration; the
short form is that the 114-column layout DROPS `scaled_state`, so re-laying-out the teacher's
input projection into it costs exactly this and nothing else.

Pre-registered expectation: a SMALL loss, under 0.002 on imm. Over ~0.004 means the teacher is
materially crippled and the re-lay-out should not be used.

⚠ This bounds the TEACHER's degradation, not the KD gain that survives it. KD pays here through
target-variance reduction, which is not linear in teacher quality -- a small number licenses
building the arm, it does not predict its size.
"""
import json
import os
import sys

A, B = "t114a", "t114b"


def load(tag, mode):
    f = "result/RWKV-%s.jsonl" % tag if mode == "ahead" else "result/RWKV-P-%s.jsonl" % tag
    if not os.path.exists(f):
        return None
    out = {}
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["user"]] = r
    return out


def main():
    print("=" * 74)
    print("TEACHER-114 SCREEN -- cost to the d=128 teacher of losing `scaled_state`")
    print("=" * 74)
    rows = {}
    for tag in (A, B):
        for mode in ("ahead", "imm"):
            d = load(tag, mode)
            if d is None:
                print("MISSING result for %s/%s -- arms have not both run." % (tag, mode))
                return 2
            rows[(tag, mode)] = d

    users = sorted(set(rows[(A, "ahead")]) & set(rows[(B, "ahead")]))
    if not users:
        print("*** no overlapping users -- nothing compared. FAILED, not a pass.")
        return 2

    # size must be identical: same db, same filter, same users. If it is not, the arms differ in
    # something other than the mask and the delta is not attributable.
    bad = [u for u in users if rows[(A, "ahead")][u]["size"] != rows[(B, "ahead")][u]["size"]]
    print("users: %d   size mismatches: %d %s"
          % (len(users), len(bad), "(expected 0)" if not bad else "*** the arms differ in more than the mask"))

    print()
    print("  %-6s %10s %10s %12s" % ("mode", "arm A", "arm B", "cost of B"))
    out = {}
    for mode in ("ahead", "imm"):
        a = sum(rows[(A, mode)][u]["metrics"]["LogLoss"] for u in users) / len(users)
        b = sum(rows[(B, mode)][u]["metrics"]["LogLoss"] for u in users) / len(users)
        out[mode] = b - a
        print("  %-6s %10.6f %10.6f %+12.6f" % (mode, a, b, b - a))

    print()
    imm = out["imm"]
    if imm < 0.002:
        print("=> SMALL (imm %+.6f < 0.002, as pre-registered). The teacher survives the" % imm)
        print("   re-lay-out: copy its 92 input columns to their names in the 114 layout, zero")
        print("   the new ones, regenerate the dump, and the features phase KEEPS KD.")
    elif imm < 0.004:
        print("=> MODERATE (imm %+.6f). Above the pre-registered 'small' bar but below the" % imm)
        print("   0.004 abort line. Worth doing only if the KD gain it protects (~0.0019) is")
        print("   still plausibly larger -- which this measurement does NOT establish.")
    else:
        print("=> LARGE (imm %+.6f >= 0.004). The teacher is materially crippled without" % imm)
        print("   `scaled_state`. Do NOT re-lay-it-out; either train KD-off or find another")
        print("   teacher that natively accepts 114 dims.")
    print()
    print("REMINDER: this bounds the TEACHER's degradation, not the KD gain that survives it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
