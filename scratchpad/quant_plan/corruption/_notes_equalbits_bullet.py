"""One-shot (2026-11-21 23:25): the NOTES.md Live-status bullet for chain 11 `it36equalbits`.
Refuses to run twice."""
import pathlib

P = pathlib.Path(__file__).resolve().parents[1] / "NOTES.md"

BULLET = """* 2026-11-21 23:25 - **Chain 11 `it36equalbits` is queued behind `it36corrupt`, so the GPU does not idle when the current queue drains.** The chain takes the equal-bits question rows 9/23 settled on the card (rank beats precision at a fixed bit count) to the three places the running ladder leaves open, four 124-step decay points all scored against row 0: **`l3i4s4`** (card rank-3 int4 + int4 shifts, note at row 17's rank-1 int8 + int4 shifts, 1,024/2,304 = **3,328**) - the cheapest card that can pass, since `l2i4s4` was only +0.000580 imm at 768 bits and one more rank has room to close the 0.00008 gap; **`l2i8s2`** (card rank-2 int8 + **int2** shifts, 1,152/2,304 = 3,456) - the card's own shift vectors one level below the free int4 (rows 9/10); and the equal-2,112-bit note pair **`n1i8s3`** (rank-1 int8 + int3 shifts) and **`n2i4s3`** (rank-2 int4 + int3 shifts), 1,280/2,112 = 3,392 each - int3 is the half-step between the working int4 shifts and the int2 shifts that cost row 20 +0.000408 imm, and the pair asks the note-side rank-vs-precision question directly. All four are inside the bit bars (card 1,024-1,280 of 2,176, note 2,112-2,304 of 6,528) and need no engine work beyond what row 15 already proved (`card:2:int8` parses; rank 2 on the card measured 0.812x). Because it is queued behind `it36corrupt`, its `BUSY_PAT` adds `probe_get_result|check_corruption` - the exact trap `QUEUE.md` warns about - and `it36equalbits` is now in `QUEUE.md`'s LIVE list so the hourly supervisor may restart it. Also corrected a stale `AGENTS.md` claim in the same pass: **rank >= 2 is deployable on the card** (0.812x), only a rank >= 2 note would need engine work; the 0.629x figure was a both-streams measurement.

"""

with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()
if "Chain 11 `it36equalbits`" in text:
    raise SystemExit("bullet already present")
for nl in ("\n", "\r\n"):
    anchor = f"## Live status{nl}{nl}"
    if anchor in text:
        text = text.replace(anchor, anchor + BULLET.replace("\n", nl), 1)
        break
else:
    raise SystemExit("Live status anchor not found")
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("bullet inserted into", P)
