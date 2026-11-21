"""One-shot (2026-11-21 23:55): append QUEUE.md section 7 (what to do when a corruption verdict FAILs,
Andrew's instruction 3: consult the advisor). Refuses to run twice."""
import pathlib

P = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\autorun\QUEUE.md")
BLOCK = """
## 7. If a corruption verdict is FAIL

Andrew's gate (2026-11-21 23:05): a non-finite card/note state value, or a non-finite `p(recall)`, makes
the recipe undeployable. The fix (a state-growth penalty plus a max cap, or a re-normalization) and the
advisor consult that Andrew asked for are the interactive session's job, not yours.

So: record the verdict exactly as section 4 says, then log
`NEEDS ANDREW: corruption FAIL for <tag> - state-growth fix + advisor consult required` and STOP.
Do NOT change the quantizer or any other source file, do NOT restart the chain that produced the
verdict, do NOT start anything, and do NOT describe the row as passing anywhere.
"""

with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()
if "## 7. If a corruption verdict is FAIL" in text:
    raise SystemExit("section 7 already present")
if not text.endswith("\n"):
    text += "\n"
text += BLOCK.replace("\n", "\r\n") if "\r\n" in text else BLOCK
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("section 7 appended to", P)
