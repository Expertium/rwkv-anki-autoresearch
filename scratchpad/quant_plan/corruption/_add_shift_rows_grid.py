"""Add the rows the PER-STREAM shift level unblocks to rust_grid.CONFIGS.

`n1i8s3` is the important one: it is the SMALLEST row that passes the loss gate (3,392 bits vs row 17's
3,584) and it was loss-only purely because the engine could not express "card int4 shifts, note int3
shifts" - a bare level was all it took, and a mixed string was silently ignored. model.rs now has a
per-stream `shift_level`, so this row can finally be priced on the deploy speed axis (the gate needs
>= 0.75x rev/s). Rows 31 (`l2i8n2i8s8`) and 34 (`l2i8s2`) become scorable for the record as well.
"""
from pathlib import Path

GRID = Path(__file__).resolve().parents[3] / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR = '''    # FIXED-BASIS (codec) state coding (approach 1, 2026-11-22):'''

NEW = '''    # THE PER-STREAM SHIFT LEVEL UNBLOCKS THESE (2026-11-22). Until model.rs grew a per-stream
    # `shift_level`, the engine took ONE bare level and ignored "card:int4,note:int3", so any row whose
    # two streams want different shift widths could not be run at all (recorded as "loss-only" in
    # RESULTS.md). `n1i8s3` is the SMALLEST row that PASSES the loss gate:
    #   card rank-2 int8, int4 shifts = 2*2*32*8 + 64*4 = 1,280
    #   note rank-1 int8, int3 shifts = 3*(2*1*32*8 + 64*3) = 2,112     total 3,392 (192 BELOW row 17)
    # It is otherwise row 17's exact layout, so its deploy speed should land at row 17's ~0.80x.
    "n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},
    # row 31 (l2i8n2i8s8, 5,888) and row 34 (l2i8s2, FAIL on the loss gate) - scored for the record.
    # Row 31 needs the fast rank-r solver for its rank-2 note (E2), hence RWKV_LOWRANK_FAST.
    "l2i8n2i8s8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                   "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int8", "RWKV_LOWRANK_FAST": "1"},
    "l2i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",
               "RWKV_STATE_SHIFT_LEVEL": "card:int2,note:int4"},

    # FIXED-BASIS (codec) state coding (approach 1, 2026-11-22):'''

raw = GRID.read_text(encoding="utf-8")
if '"n1i8s3"' in raw:
    raise SystemExit("rust_grid.py already has n1i8s3")
n = raw.count(ANCHOR)
if n != 1:
    raise SystemExit(f"fb-block anchor matched {n} times (need 1)")
GRID.write_text(raw.replace(ANCHOR, NEW, 1), encoding="utf-8")
print(f"patched {GRID}")
