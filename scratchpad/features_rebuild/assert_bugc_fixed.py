"""Phase-0 guard: is the Bug C fix actually LIVE before we spend ~4 h rebuilding for it?

WHY THIS EXISTS. Generation 4 has exactly one purpose -- carry the Bug C fix into the -id
databases. A build that silently reproduced Bug C would be worthless AND invisible: every banner,
every width check and every entry count would pass, because Bug C destroys VALUES, not shapes.
That is the same failure shape as the QAT env that was parsed and then discarded, and as the
featA2 number retracted on 2026-09-01.

WHAT BUG C IS. A NaN note_id is replaced by `ID_PLACEHOLDER + card_id` so each such card gets its
OWN note. ID_PLACEHOLDER is 3.14e17, past float64's exact-integer limit of 2^53 where the spacing
is 64 -- so doing that arithmetic in a float64 column silently merges any two cards whose ids are
within 64. Measured: published 49,186 -> 812 distinct placeholders (98.3% lost), -id 49,186 ->
30,869 (37.2% lost).

THE GUARD PROVES ITS OWN NON-VACUITY. It re-runs the check against a simulated float64 path and
REQUIRES that to collapse. Without that, the test would pass just as happily on a build where
nothing works, which is precisely the class of green this project keeps having to retract.

Exit 0 = fix is live. Exit 45 = fix is NOT live. Exit 46 = the test itself is vacuous.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rwkv.data_processing import ID_PLACEHOLDER, nan_id_fill  # noqa: E402

# Real -id card ids are raw Anki epoch-ms, ~1.7e12. Spaced 1 ms apart is the worst realistic
# case and the one float64 destroys: at 3.14e17 the representable spacing is 64.
BASE = 1_708_127_478_116
CARD_IDS = np.array([BASE + i for i in range(4096)], dtype=np.int64)


def main():
    ok = True

    # ---- 1. the live implementation must be injective on distinct card ids ----
    live = nan_id_fill("note_id", CARD_IDS)
    n_live = len(set(int(x) for x in live))
    print("live nan_id_fill : %d distinct placeholders for %d distinct card ids"
          % (n_live, CARD_IDS.size))
    if n_live != CARD_IDS.size:
        print("  *** BUG C IS LIVE -- the fix is NOT in effect. Do not rebuild.")
        ok = False

    # ---- 2. exactness, not just distinctness ----
    expect = np.int64(ID_PLACEHOLDER) + CARD_IDS
    if not np.array_equal(live, expect):
        print("  *** placeholders are distinct but not EXACT (%d mismatches)"
              % int((live != expect).sum()))
        ok = False
    else:
        print("live nan_id_fill : exact against int64 ID_PLACEHOLDER + card_id")

    # ---- 3. deck/preset must stay POOLED -- a constant, deliberately ----
    for name in ("deck_id", "preset_id"):
        v = nan_id_fill(name, CARD_IDS)
        if len(set(int(x) for x in v)) != 1 or int(v[0]) != ID_PLACEHOLDER:
            print("  *** %s should be the bare constant (deliberate pooling), got %d distinct"
                  % (name, len(set(int(x) for x in v))))
            ok = False
    print("live nan_id_fill : deck_id/preset_id remain the bare constant (pooled by design)")

    # ---- 4. NON-VACUITY: the old float64 path MUST collapse, or this test proves nothing ----
    buggy = (np.float64(ID_PLACEHOLDER) + CARD_IDS.astype(np.float64)).astype(np.int64)
    n_buggy = len(set(int(x) for x in buggy))
    print("simulated float64: %d distinct placeholders (Bug C's behaviour)" % n_buggy)
    if n_buggy >= CARD_IDS.size:
        print("  *** the float64 simulation did NOT collapse, so this guard cannot detect Bug C.")
        print("      The test is VACUOUS -- fix the test before trusting any build.")
        return 46

    print()
    if ok:
        print("BUG C FIX IS LIVE (and the guard is provably able to detect its absence).")
        return 0
    print("BUG C FIX IS NOT LIVE -- refusing the rebuild.")
    return 45


if __name__ == "__main__":
    raise SystemExit(main())
