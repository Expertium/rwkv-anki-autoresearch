"""Composition of the candidate 100k designs: where does each design SPEND its budget?

The point is the REBALANCE Andrew asked for -- "allocate >=99% of parameters to how other
input features are processed". This prints the feature-MLP share against the stream share,
so the trade is visible instead of asserted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "scratchpad", "hybrid100k"))
from budget import price, CHAMP_STRIP

CANDS = [
    ("champion d=80 13L fm4", dict(strip_cmix=CHAMP_STRIP)),
    ("d=32 ctx 1/2/2/2 fm4", dict(head_dim=16, n_heads=2, L=(2,1,2,2,2), feat_mult=4,
                                  strip_cmix=CHAMP_STRIP)),
    ("d=32 ctx 1/2/2/2 fm8", dict(head_dim=16, n_heads=2, L=(2,1,2,2,2), feat_mult=8,
                                  strip_cmix=CHAMP_STRIP)),
    ("d=32 ctx 1/1/1/1 fm12", dict(head_dim=16, n_heads=2, L=(2,1,1,1,1), feat_mult=12,
                                   strip_cmix=CHAMP_STRIP)),
    ("d=24 ctx 1/2/2/2 fm16", dict(head_dim=12, n_heads=2, L=(2,1,2,2,2), feat_mult=16,
                                   strip_cmix=CHAMP_STRIP)),
]
print("%-24s %8s %7s %7s %7s %7s %7s %8s %6s"
      % ("design", "total", "card", "note", "deck", "preset", "user", "feat+hd", "cardSt"))
print("-" * 92)
for nm, kw in CANDS:
    d = price(nm, **kw)
    if "error" in d:
        print("%-24s ERROR %s" % (nm, d["error"])); continue
    b = d["by"]
    print("%-24s %8d %7d %7d %7d %7d %7d %8d %6d"
          % (nm, d["total"], b.get("card_id", 0), b.get("note_id", 0), b.get("deck_id", 0),
             b.get("preset_id", 0), b.get("user_id", 0), b.get("(model)", 0), d["card_state"]))
print()
print("feat+hd = input feature MLP + all output heads (the non-recurrent half).")
