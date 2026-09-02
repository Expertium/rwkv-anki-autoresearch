"""Does `ahead` use the new timestamp features at all, and which group carries them?

featB's own per-user results are the control -- same checkpoint, same db, same env; the arms
differ only in which columns are zeroed at the input. So the comparison is restricted to the
users the arms actually scored.

TWO QUESTIONS, and the second is the one that steers feature work:

  1. RELIANCE BY MODE. If ahead barely degrades when all 23 columns are removed, then ahead is
     not using them, and its small gain is a property of the model rather than of the features.
     If it degrades substantially, ahead DOES use them and the 5:1 imm-to-ahead ratio is about
     what the features can inform, not about whether they reach the ahead path.

  2. WHICH GROUP. featB's verdict refuted P2 -- the gain was NOT concentrated in same-day users,
     which pointed AWAY from the fine-grained clock columns and towards the always-defined ones.
     If that reading is right, `struct` costs more than `clock`. If it is backwards, the same-day
     analysis was measuring something else and must not steer feature work.

⚠ AN INFERENCE-TIME ABLATION MEASURES RELIANCE, NOT VALUE. A retrained model recovers part of it
-- the delta-rule caveat (zeroing `a` cost +0.208 imm on a model that had co-adapted to it) and
the teacher-114 caveat. Read this as "does ahead use these", never as "this is what they are
worth".
"""
import json
import os

CONTROL = "featB"
ARMS = [("abl_all", "all 23 new columns"),
        ("abl_clock", "the 10 fine-grained TIMING columns"),
        ("abl_struct", "the 13 always-defined STRUCTURE columns")]


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
    ctrl = {m: load(CONTROL, m) for m in ("ahead", "imm")}
    if any(v is None for v in ctrl.values()):
        print("MISSING featB results -- the control is not available.")
        return 2

    print("=" * 78)
    print("FEATURE ABLATION on featB's own checkpoint -- reliance, not value")
    print("=" * 78)

    rows = {}
    for tag, _ in ARMS:
        for mode in ("ahead", "imm"):
            d = load(tag, mode)
            if d is None:
                print("MISSING arm %s/%s -- not all arms have run." % (tag, mode))
                return 2
            rows[(tag, mode)] = d

    users = sorted(set(rows[("abl_all", "ahead")]) & set(ctrl["ahead"]))
    if not users:
        print("*** no overlapping users -- nothing compared. FAILED, not a pass.")
        return 2
    bad = [u for u in users if rows[("abl_all", "ahead")][u]["size"] != ctrl["ahead"][u]["size"]]
    print("users %d   size mismatches vs control: %d %s"
          % (len(users), len(bad), "" if not bad else "*** arms differ in more than the mask"))

    base = {m: sum(ctrl[m][u]["metrics"]["LogLoss"] for u in users) / len(users)
            for m in ("ahead", "imm")}
    print("control (featB, same users): ahead %.6f   imm %.6f" % (base["ahead"], base["imm"]))
    print()
    print("  %-12s %-40s %12s %12s" % ("arm", "what is removed", "ahead cost", "imm cost"))
    cost = {}
    for tag, desc in ARMS:
        c = {}
        for mode in ("ahead", "imm"):
            v = sum(rows[(tag, mode)][u]["metrics"]["LogLoss"] for u in users) / len(users)
            c[mode] = v - base[mode]
        cost[tag] = c
        print("  %-12s %-40s %+12.6f %+12.6f" % (tag, desc, c["ahead"], c["imm"]))

    print()
    a = cost["abl_all"]
    print("--- Q1: does `ahead` use the new features at all?")
    if a["ahead"] < 0.0005:
        print("    NO, essentially. Removing all 23 costs ahead %+.6f -- under the 0.0005 that one" % a["ahead"])
        print("    accepted iteration is worth. ahead's small gain is a property of the MODEL, not")
        print("    of the features: they are reaching it and it is barely using them.")
    else:
        print("    YES. Removing all 23 costs ahead %+.6f, so the features ARE load-bearing there." % a["ahead"])
        print("    The 5:1 imm-to-ahead ratio is then about what these columns can INFORM -- they")
        print("    describe `now`, which is what imm predicts -- not about reach.")
    if a["imm"] > 0:
        print("    ratio imm:ahead reliance = %.2f:1" % (a["imm"] / a["ahead"]) if a["ahead"] > 0
              else "    ahead cost is <= 0; ratio undefined")

    print()
    print("--- Q2: which group carries it? (featB's P2 refutation predicts STRUCT > CLOCK)")
    ck, st = cost["abl_clock"], cost["abl_struct"]
    for mode in ("ahead", "imm"):
        verdict = ("STRUCT dominates, as the same-day analysis implied"
                   if st[mode] > ck[mode] else
                   "CLOCK dominates -- this CONTRADICTS the same-day reading")
        print("    %-6s clock %+.6f   struct %+.6f   -> %s" % (mode, ck[mode], st[mode], verdict))
    print()
    print("REMINDER: reliance, not value. A retrained model recovers part of any of these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
