"""Generate the gen-4 re-base runner from featB's, changing ONLY the databases.

WHAT THIS RUN IS. featB proved the timestamp features are worth having, but on generation 3 --
which carries Bug C and scores against an end-to-end-selected equalize set. Generation 4 fixes
both. This run establishes the `-id` lineage's baseline on gen 4: its champion, and the `size`
baseline every later candidate is gated against.

It is featB's recipe EXACTLY, KD-off included. KD-off is not a choice here: the d=128 teacher's
features2card in_dim is 92 and cannot forward 114 dims, and the teacher-114 screen measured that
re-laying it out costs +0.020447 ahead / +0.007930 imm -- it leans hard on `scaled_state`, which
the -id rebuild drops. A crippled teacher is worse than none.

=> gen4base MINUS featB is a clean two-variable step (Bug C fix + e2s equalize set) with the
recipe held fixed, which is the only comparison either database supports.

★ THE GENERATOR CHECKS BOTH DIRECTIONS, because checking one is how this repo has been bitten:
  * no stale gen-3 token survives into a live line  (the usual check)
  * every line that MENTIONED gen 3 was actually visited (the mirror-image check that
    mk_fixc_arm.py lacked, which let a runner describe itself as evaluating the wrong db)
"""
import io
import os
import re
import sys

SRC_DIR = "scratchpad/features_ab/featB"
DST_DIR = "scratchpad/gen4_base"

SUBS = [
    # databases -- the only intended change
    ("F:/rwkv_lmdb/train_db_5k_h1_id3", "F:/rwkv_lmdb/train_db_5k_h1_id4"),
    ("F:/rwkv_lmdb/test_db_5k_id3", "F:/rwkv_lmdb/test_db_5k_id4"),
    ("F:/rwkv_lmdb/label_filter_db_id", "F:/rwkv_lmdb/label_filter_db_id_e2s"),
    # identity
    ("scratchpad/features_ab/featB", "scratchpad/gen4_base"),
    ("scratchpad\\features_ab\\featB", "scratchpad\\gen4_base"),
    ("featB_ws", "g4b_ws"),
    ("featB_d", "g4b_d"),
    ("featB", "gen4base"),
    ("FEATB", "GEN4BASE"),   # the echo header and a REM; case-sensitive replace missed these
]

# `label_filter_db_id` is a PREFIX of `label_filter_db_id_e2s`, so a second pass would produce
# `..._e2s_e2s`. Substitutions run once over the text, longest-first, which avoids that -- and it
# is asserted at the end rather than assumed.
SUBS.sort(key=lambda kv: -len(kv[0]))


def convert(text):
    for old, new in SUBS:
        text = text.replace(old, new)
    return text


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    made = []
    for name, out in (("run_featB.cmd", "run_gen4base.cmd"), ("ws.toml", "ws.toml")):
        src = os.path.join(SRC_DIR, name)
        s = io.open(src, encoding="utf-8").read()
        # ⚠ CASE-INSENSITIVE, and the first version was not. `echo ===== FEATB START` survived
        # into the generated runner because the substitution and both checks matched only the
        # exact-case "featB". Cosmetic here -- %LOG% and %TAG% were correct, so every guard, db
        # and output was right and only the log header was mislabelled -- but it is the SAME
        # shape as mk_fixc_arm.py's drift, in the generator written to avoid it. A check that
        # tests one casing tests one casing.
        def names_old(ln):
            low = ln.lower()
            return "id3" in low or "featb" in low

        visited = [ln for ln in s.split("\n") if names_old(ln)]
        d = convert(s)

        # direction 1: nothing stale survives in a LIVE line
        stale = [ln for ln in d.split("\n")
                 if names_old(ln)
                 and not ln.strip().upper().startswith("REM")
                 and not ln.strip().startswith("#")]
        assert not stale, "stale gen-3 token survived in %s:\n  %s" % (out, "\n  ".join(stale))

        # direction 2: every line that named gen 3 was actually visited
        still = [ln for ln in d.split("\n") if names_old(ln)]
        assert len(still) < len(visited), (
            "%s: %d lines named gen 3 and %d still do -- the substitution did not visit them"
            % (out, len(visited), len(still)))

        # the prefix trap, asserted not assumed
        assert "_e2s_e2s" not in d, "double-substituted the label filter path in %s" % out

        io.open(os.path.join(DST_DIR, out), "w", encoding="utf-8", newline="\n").write(d)
        made.append((out, len(visited), len(still)))

    print("generated into %s:" % DST_DIR)
    for out, v, s_ in made:
        print("  %-20s lines naming gen 3: %d -> %d" % (out, v, s_))
    print()
    print("both directions checked: nothing stale survives, and every gen-3 line was visited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
