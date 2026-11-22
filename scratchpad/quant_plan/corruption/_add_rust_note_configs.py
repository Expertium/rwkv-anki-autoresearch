"""Add the rank-2/3-note configs to rust_grid.CONFIGS (the deploy engine's speed grid)."""
from pathlib import Path

GRID = Path(__file__).resolve().parents[1] / "rust_grid.py"
text = GRID.open(encoding="utf-8", newline="").read()

if '"l2i8n2i8"' in text:
    raise SystemExit("configs already present")

NEW = '''    # Rank-2 (and 3) on the NOTE, the new chain-14 candidates (Andrew, 2026-11-22 12:01). The engine's
    # low-rank path is generic over the stream, so these exist to prove it parses a rank>=2 note and to
    # price the speed. l2i8n2i8 is the direct twin of l2i8s4 with the note's rank raised 1 -> 2.
    "l2i8n2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4"},
    "l3i8n2i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:3:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4"},
    "l2i8n2i8s8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:2:int8", "RWKV_QUANT_SHIFTS": "1",
                   "RWKV_STATE_SHIFT_LEVEL": "note:int8,card:int4"},
    "l2i8n3i8": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:3:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_SHIFT_LEVEL": "int4"},
'''

anchor = '    "l1i4als": {"RWKV_STATE_LOWRANK_SCOPE": "card:1:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",\n                "RWKV_LOWRANK_ALS": "3"},'
if anchor not in text:
    # tolerate CRLF
    anchor = anchor.replace("\n", "\r\n")
    if anchor not in text:
        raise SystemExit("anchor (l1i4als) not found")

nl = "\r\n" if anchor.endswith("\r\n") else "\n"
text = text.replace(anchor, anchor + nl + NEW.replace("\n", nl), 1)

with GRID.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("added 4 engine configs")
