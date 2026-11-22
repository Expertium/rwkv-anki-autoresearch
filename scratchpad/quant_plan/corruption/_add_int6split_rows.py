"""Add the two MIXED-precision int6 attribution configs to rust_grid.CONFIGS.

Chain 17's ladder coarsens card and note together (`card:2:int6,note:1:int6`), so a failure cannot say
WHERE the int6 tax comes from. These two hold one stream at int8 and coarsen the other, at row 17's
layout, so a deploy-side screen can split the tax:

    l2i6c  card rank-2 int6 + note rank-1 int8 = 1,024 + 2,304 = 3,328 (256 BELOW row 17)
    l2i6n  card rank-2 int8 + note rank-1 int6 = 1,280 + 1,920 = 3,200 (384 BELOW row 17)

Both are PTQ screens on the row-17 deploy weights (the same method as the int6/int5 screen). Raw bytes
with per-anchor newline detection: the CONFIGS block is LF inside a mixed-newline file.
"""
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
GRID = RUN / "rust_grid.py"

ANCHOR = (
    '    "l2i8n3i8": {{"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:3:int8", "RWKV_QUANT_SHIFTS": "1",{nl}'
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"}},{nl}'
)

NEW = (
    '{nl}'
    '    # MIXED factor precision (2026-11-22): chain 17 coarsens BOTH streams at once'
    ' (`card:2:int6,note:1:int6`),{nl}'
    '    # so it cannot say WHERE the int6 tax comes from. These hold one stream at int8 and coarsen the'
    ' other,{nl}'
    '    # at row 17\'s layout: card int6 + note int8 = 1,024 + 2,304 = 3,328 (256 BELOW row 17), and'
    ' card int8{nl}'
    '    # + note int6 = 1,280 + 1,920 = 3,200 (384 BELOW row 17). Attribution screens, not candidates.'
    '{nl}'
    '    "l2i6c": {{"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8", "RWKV_QUANT_SHIFTS": "1",{nl}'
    '              "RWKV_STATE_SHIFT_LEVEL": "int4"}},{nl}'
    '    "l2i6n": {{"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int6", "RWKV_QUANT_SHIFTS": "1",{nl}'
    '              "RWKV_STATE_SHIFT_LEVEL": "int4"}},{nl}'
)


def main() -> int:
    raw = GRID.read_bytes()
    if b'"l2i6c"' in raw:
        print("already present")
        return 0
    for nl in ("\r\n", "\n"):
        ob = ANCHOR.format(nl=nl).encode("utf-8")
        if raw.count(ob) != 1:
            continue
        raw = raw.replace(ob, ob + NEW.format(nl=nl).encode("utf-8"), 1)
        GRID.write_bytes(raw)
        print(f"patched rust_grid.py (newline={nl!r})")
        return 0
    raise SystemExit("ABORT: no single-newline match for the l2i8n3i8 anchor")


if __name__ == "__main__":
    sys.exit(main())
