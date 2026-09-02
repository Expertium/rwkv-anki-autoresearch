"""Ablate input dims by CHECKPOINT SURGERY: zero columns of `features2card.0.weight`.

WHY SURGERY AND NOT A FLAG. The two masks the model offers cannot reach the encoding block under
RWKV_ID_FEATURES=1: `RWKV_ZERO_FEATURES` (index-based) is refused there by design, and
`RWKV_ABLATE_FEATURES` resolves NAMES through CARD_FEATURE_COLUMNS, i.e. dims 0..45 only. Adding a
third mask means editing srs_model.py -- which gen4base's decay and eval phases will re-import,
the mid-chain import trap. Zeroing the weight column instead touches no code the run loads.

WHY IT IS EXACT. The input projection is `y = W x + b`, linear in x, so zeroing column j of W is
identical to zeroing feature j of every input row. model.rs relies on the same identity to apply
RWKV_ZERO_FEATURES at load; this is the Python-side twin of that trick.

NON-VACUITY. The script REFUSES to write a checkpoint whose target columns were already zero, and
reports the pre-surgery norm of every range it zeroes. An ablation of columns that carried nothing
would produce a candidate identical to the control and a clean null that reads as "these dims do
not matter" when it means "nothing was removed" -- the false-green shape this repo keeps paying for.

Usage: zero_input_cols.py <in.pth> <out.pth> <lo:hi> [<lo:hi> ...]     (hi exclusive)
"""
import sys

import torch

KEY = "features2card.0.weight"


def main():
    src, dst, ranges = sys.argv[1], sys.argv[2], sys.argv[3:]
    sd = torch.load(src, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    w = sd[KEY]
    out_dim, in_dim = w.shape
    print("[surgery] %s  %s  shape (%d, %d)" % (src, KEY, out_dim, in_dim))

    total_cols = 0
    for r in ranges:
        lo, hi = (int(t) for t in r.split(":"))
        assert 0 <= lo < hi <= in_dim, "range %s outside [0, %d)" % (r, in_dim)
        block = w[:, lo:hi]
        norm_before = float(block.norm())
        if norm_before == 0.0:
            print("[surgery] *** columns %d:%d are ALREADY ZERO -- ablating them removes nothing." % (lo, hi))
            print("[surgery]     Refusing to write: the arm would be a vacuous control. Not a pass.")
            return 47
        w[:, lo:hi] = 0.0
        total_cols += hi - lo
        print("[surgery] zeroed columns %d:%d  (%d cols, pre-surgery Frobenius norm %.4f)"
              % (lo, hi, hi - lo, norm_before))

    torch.save(sd, dst)
    print("[surgery] wrote %s  -- zeroed %d input columns in total" % (dst, total_cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
