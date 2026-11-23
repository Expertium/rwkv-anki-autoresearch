"""Verify the anchors _record_row44.py will use, before it writes anything."""
from pathlib import Path

RUN = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\optimization\GPU_RUN_BOARD.md")
RECORDED = RUN / "autorun" / "RECORDED.txt"

res = RESULTS.read_bytes()
nb = NOTES.read_bytes()
bb = BOARD.read_bytes()
rb = RECORDED.read_bytes()

A_ROW = b"so the whole int4-card ladder fails whatever note sits under it | - |"
A_PARA = b"this screen is PTQ against weights trained for the rank-1-int8 note, so it cannot see what QAT recovers."
A_STAT = b"corruption included**"
A_NOTE = b"## Live status\n\n"

print("ROW  ", res.count(A_ROW))
print("PARA ", res.count(A_PARA))
print("STAT ", res.count(A_STAT))
print("NOTE ", nb.count(A_NOTE))
print("BOARD has Amd 28:", b"## Amendment 28" in bb)
print("BOARD has Amd 29:", b"## Amendment 29" in bb)
print("RECORDED has it36l2i8s4pf:", b"it36l2i8s4pf" in rb.split(b"\n"))
print("RESULTS has row 44:", b"| 44 |" in res)
print("NOTES tail:")
print(nb.decode("utf-8", "replace").split("## Live status")[1][:600])
