"""Append board amendment 11: row 15 is the first passing recipe."""
from pathlib import Path

P = Path("optimization/GPU_RUN_BOARD.md")
blk = """
Row 11 amendment 11 (2026-11-21 08:40, Moscow). **ROW 15 PASSES ALL SIX CONSTRAINTS - the card/note state
quantization arm has its first passing recipe.** `l2i8` = QAT card RANK-2 int8 factors + int8 shifts, note
rank-1 int8: card 1,536 bits (192 B, 2.83x smaller than naive int4, bar 2,176), note 3,072 bits (384 B,
2.13x smaller, bar 6,528). Loss on the 750-user roster = 0.333848/0.306518, i.e. +0.000046 ahead (ub95
+0.000112) and +0.000307 imm (ub95 +0.000369) against the identical unquantized 124-step decay (row 0) -
both heads inside the +0.0005 ceiling including the upper bounds. Deploy engine speed 2,323.6 rev/s =
0.812x (states/s 0.660x, median per-user 0.803x), above the 0.75x floor, and that measurement was about
3% pessimistic because a QAT decay was training (`l1i8` read 0.875x there against 0.903x idle). The
mechanism is confirmed: rank-1 imm +0.001009 -> rank-2 +0.000307. The earlier "rank >= 2 cannot be
deployed" conclusion came from `l2i4`, which put rank 2 on the NOTE as well (4 layers) and measured
0.629x; card-only rank-2 is 0.812x and needs NO engine change, because the engine already parses
`card:2:int8`. Artifacts: `rust_artifacts/it36l2i8_124.safetensors`, `rust_speed_rank2card.json`, and the
3-user deploy grid `rust_grid_it36l2i8.json`. OWED: an idle-machine speed re-measurement, and Andrew's
decision on whether this recipe replaces the deployed 403/214-bit one. STILL RUNNING: rows 16-18 (card
rank-3, int4 shifts) and `it36notecut` behind them. GPU at amendment time: `it36rank`'s `l3i8` QAT decay.
"""
with P.open("ab") as fh:
    fh.write(blk.encode("utf-8"))
text = P.read_text(encoding="utf-8-sig")
print("amendment 11 markers:", text.count("Row 11 amendment 11"))
print("board lines:", text.count("\n") + 1)
