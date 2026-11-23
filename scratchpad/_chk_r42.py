import json
from pathlib import Path

RUN = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
res = (RUN / "RESULTS.md").read_bytes()
notes = (RUN / "NOTES.md").read_bytes()

print("ROW  ", res.count(b"this exact 3,136 bits) tests | - |"))
print("PARA ", res.count(b"unless QAT recovers most of a +0.0020/+0.0028 gap."))
print("STAT ", res.count(b"rows 33-41 FAIL too; row 35 (`n1i8s3`) PASSES** (row 41 "))
print("NOTE ", notes.count(b"## Live status\n\n"))
print("BUL  ", notes.count(b"Row 42 (`l3i4s4n3`) closes"))
print("MARK ", res.count(b"**Row 42 (`l3i4s4n3`) closes the 2,368-bit rung"))

s = res.decode("utf-8", "replace")
for line in s.splitlines():
    if line.startswith("| 33 |") or line.startswith("| 41 |"):
        print(line[:420])
        print("---")
