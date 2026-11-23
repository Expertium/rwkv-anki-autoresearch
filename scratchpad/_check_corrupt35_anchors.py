from pathlib import Path

RUN = Path("scratchpad/quant_plan")
res = (RUN / "RESULTS.md").read_bytes()
notes = (RUN / "NOTES.md").read_bytes()
board = Path("optimization/GPU_RUN_BOARD.md").read_bytes()

A_PEND = (b"`pending` means the chain has not reached the row yet; row 35 (`n1i8s3`) is the only `pending` "
          b"cell now, because `it36corrupt2` reaches it first (its earlier points "
          b"`l2i4s4`/`l3i4s4`/`l2i8s2` are gate-skipped - rows 23, 33 and 34 fail the loss gate).")
A_STAT = b"row 35 (`n1i8s3`) PASSES**"
A_NOTE = b"## Live status\n\n"
print("A_PEND", res.count(A_PEND))
print("A_STAT", res.count(A_STAT))
print("A_NOTE", notes.count(A_NOTE))
print("amend27", b"Amendment 27" in board)
print("notes bullet already", notes.count(b"(row 35) PASSES the corruption gate"))