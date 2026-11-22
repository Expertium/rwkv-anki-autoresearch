"""Fix the rank-2-note engine configs: the engine's parse_level() accepts only a BARE level, so
per-stream shift strings ("note:int8,card:int4") silently mean "no override". Use bare levels."""
from pathlib import Path

GRID = Path(__file__).resolve().parents[1] / "rust_grid.py"
text = GRID.open(encoding="utf-8", newline="").read()

old = '''    "l2i8n2i8s8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                   "RWKV_STATE_SHIFT_LEVEL": "note:int8,card:int4"},
'''
new = '''    # NOTE: the engine has NO per-stream shift level - parse_level() takes a bare "int4"/"int8" and any
    # "card:x,note:y" string yields None (= no override). So `l2i8n2i8s8` (card int4 / note int8 shifts)
    # cannot be expressed on the deploy side yet; that is a small engine gap, recorded in GPU_RUN_BOARD.
'''
if old not in text:
    raise SystemExit("the s8 config is not in the form expected")

text = text.replace(old, new, 1)
with GRID.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("dropped l2i8n2i8s8 from the engine grid (cannot be expressed)")
