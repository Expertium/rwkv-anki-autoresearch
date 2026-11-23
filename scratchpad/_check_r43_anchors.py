from pathlib import Path

RUN = Path("scratchpad/quant_plan")
res = (RUN / "RESULTS.md").read_bytes()
notes = (RUN / "NOTES.md").read_bytes()

A_ROW = b"so the note's FACTOR width is the binding constraint a third time and 2,368 bits is not reachable on this card | - |"
A_PARA = b"unless the rung above it is proven."
A_STAT = b"rows 33-42 FAIL too; row 35 (`n1i8s3`) PASSES** (row 42 "
A_NOTE = b"## Live status\n\n"
print("row anchor", res.count(A_ROW))
print("para anchor", res.count(A_PARA))
print("stat anchor", res.count(A_STAT))
print("note anchor", notes.count(A_NOTE))
print("already row43", b"| 43 |" in res)
print("amend26", b"Amendment 26" in (Path("optimization/GPU_RUN_BOARD.md").read_bytes()))
rec = (RUN / "autorun" / "RECORDED.txt").read_text(encoding="utf-8").splitlines()
print("stamp present", "it36l2i4s4n8s3" in rec, "last3", rec[-3:])
