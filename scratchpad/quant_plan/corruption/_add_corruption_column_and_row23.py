"""One-shot (2026-11-21 23:00, Moscow).

(1) Add Andrew's requested `corruption` column to the main RESULTS.md table (line 64 header, line 65
    separator, rows 0..22).  Values: `pending` where the it36corrupt check is queued (row 0 = the fp32
    control, rows 15-18 = the four passing recipes), `-` everywhere else (the row fails the log-loss
    gate, so it is not a deploy candidate and is not checked).
(2) Record the landed sweep it36l2i4s4 as row 23 (FAIL: imm +0.000580, ub95 +0.000661).
(3) Add the corruption legend paragraph after the speed footnote.
(4) Extend the **Status:** paragraph with row 23 and the new column.

Run once; it refuses if the column is already there.
"""
import pathlib
import sys

P = pathlib.Path(__file__).resolve().parents[1] / "RESULTS.md"
raw = P.read_bytes().decode("utf-8")
if "corruption |" in raw or "| corruption" in raw:
    raise SystemExit("corruption column already present - nothing to do")

lines = raw.split("\n")  # each element keeps its own '\r' if the file used CRLF

# --- cell for every existing main-table line, keyed by 1-based line number -----------------
CELLS = {64: " corruption |", 65: "---|"}
for ln in range(66, 90):                      # rows 0, 0b, 1..22
    if ln == 66 or 82 <= ln <= 85:            # row 0 (control) and rows 15-18 (the passers)
        CELLS[ln] = " pending |"
    else:
        CELLS[ln] = " - |"

for ln, cell in CELLS.items():
    i = ln - 1
    body = lines[i]
    cr = body.endswith("\r")
    if cr:
        body = body[:-1]
    if not body.startswith("|"):
        raise SystemExit(f"line {ln} is not a table line: {body[:60]!r}")
    lines[i] = body + cell + ("\r" if cr else "")

# --- row 23: the landed it36l2i4s4 result -------------------------------------------------
ROW23 = (
    "| 23 | QAT card **rank-2 int4** factors + int4 shifts, note rank-1 int8 + int4 shifts - "
    "rank 2 at the cheap precision, the SAME 768/2,304 bits as row 9 | 768 (96) | 2,304 (288) | "
    "2,290 | 0.800x | 0.334068 | 0.306792 | +0.000266 | +0.000580 | **FAIL**, imm is 1.16x over the "
    "ceiling (ub95 +0.000337/+0.000661, p 6.9e-16/1.5e-38): at EQUAL bits rank-2 int4 beats rank-1 "
    "int8 (row 9) on both heads by about 0.00025/0.00045, so extra RANK is worth more than extra "
    "factor precision - but the imm head still needs the 2,304-bit note |"
)
END = 89  # 1-based line of row 22
cr = "\r" if lines[END - 1].endswith("\r") else ""
lines.insert(END, ROW23 + cr)

# --- legend paragraph after the speed footnote (now line 93) ------------------------------
LEGEND = (
    "**The `corruption` column (added 2026-11-21, Andrew's gate).** A row is deployable only if the "
    "quantized recipe produces NO non-finite value anywhere: if even ONE card/note state value or even "
    "ONE p(recall) is non-finite, on the users with the longest review histories and with those "
    "histories NOT truncated, the recipe FAILS. `pending` = the `it36corrupt` chain is queued for that "
    "row (the four passing rows 15-18 plus the quantization-OFF fp32 control of row 0, so the check "
    "also answers whether a blow-up needs quantization at all); `-` = not checked, because the row "
    "fails the log-loss gate and is not a deploy candidate. The verdicts land in "
    "`scratchpad/quant_plan/corruption/corruption_corrupt_<tag>.json` and are filled in here."
)
# find the last line of the speed footnote block ("... as the 750-user eval / does).")
for i in range(len(lines) - 1, 0, -1):
    if "as the 750-user eval" in lines[i]:
        lines.insert(i + 1, "")
        lines.insert(i + 2, LEGEND + ("\r" if lines[i].endswith("\r") else ""))
        break
else:
    raise SystemExit("speed-footnote anchor not found")

# --- Status paragraph --------------------------------------------------------------------
STATUS_ADD = (
    " **Row 23 (`l2i4s4`) FAILS, and it settles one axis-1 question**: putting rank 2 on the card and "
    "the card factors at int4 reaches the SAME 768 card / 2,304 note bits as row 9 and scores "
    "0.334068/0.306792 = **+0.000266 ahead / +0.000580 imm** (ub95 +0.000337/+0.000661) - better than "
    "row 9's +0.000516/+0.001030 on both heads at identical bits, so at a fixed bit budget rank beats "
    "factor precision on the card; the imm head is still 1.16x over the ceiling, so an int8-quality "
    "note (2,304 bits) remains part of the cheapest passing set. The `corruption` column is new: row 0 "
    "and rows 15-18 are queued in `it36corrupt`."
)
for i, ln in enumerate(lines):
    if ln.startswith("**Status: rows 15-18 PASS all six constraints"):
        # append to the END of that paragraph (the next blank line)
        j = i
        while j + 1 < len(lines) and lines[j + 1].strip() != "":
            j += 1
        cr2 = "\r" if lines[j].endswith("\r") else ""
        lines[j] = lines[j][:-1] + " " + STATUS_ADD if cr2 else lines[j] + " " + STATUS_ADD
        lines[j] = lines[j] + cr2
        break
else:
    raise SystemExit("Status paragraph anchor not found")

P.write_bytes("\n".join(lines).encode("utf-8"))
print("wrote", P)
