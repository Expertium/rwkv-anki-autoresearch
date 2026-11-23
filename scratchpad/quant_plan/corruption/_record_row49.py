"""Record RESULTS row 49 (`l2i6n`, chain 18 point 2) + its paragraph, a NOTES bullet and BOARD Amendment 35.

THE POINT. `it36int6card` point 1 (`l2i6c`) held the note at row 17's int8 and cut only the CARD: +0.000085
ahead / +0.000364 imm at 3,328 bits. Point 2 is the mirror image: row 17's card with the NOTE's factor
width cut int8 -> int6 (rank-1, int4 shifts), i.e. 1,280/1,920 = 3,200 bits - 384 below row 17 and 256
ABOVE row 45. Together the two rows give the two streams' int6 costs DIRECTLY, which matters because the
11-user deploy screen (`rust_wide_screen_it36int6split.json`) said the NOTE carries most of the tax
(+0.000268/+0.000307 for the note's cut vs +0.000052/+0.000115 for the card's) while row 45's all-at-once
cut could not separate them, and row 48's own difference-of-deltas reading ("proportional to the bits
they save") disagreed with that screen. This row decides between the two readings on 750 users.

WHY THIS FILE IS WRITTEN BEFORE THE SWEEP LANDS. The recorder must be ready the second the sweep file
appears (Andrew's standing order is to record and push immediately), and the verdict was not known when
it was written. So every number and every verdict word is read from `quant_sweep_it36l2i6n.json` at run
time, and the interpretation sentence picks the reading that the numbers support, instead of being
hard-coded - which also removes the main transcription risk of the row-48 recorder.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_row49.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]           # scratchpad/quant_plan/corruption -> repo root
RUN = ROOT / "scratchpad" / "quant_plan"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"
SWEEP = RUN / "quant_sweep_it36l2i6n.json"
CHAIN_LOG = RUN / "it36int6card_chain.log"

STAMP = "it36l2i6n"
CEIL = 0.0005
# The three reference rows this row is priced against, as (ahead, imm) deltas vs qnj_it36ctl (750 users)
ROW17 = (0.000052, 0.000327)   # both streams int8 + int4 shifts, 3,584 bits
ROW45 = (0.000125, 0.000381)   # both streams int6, 2,944 bits
ROW48 = (0.000085, 0.000364)   # card int6 + note int8, 3,328 bits (this chain's point 1)
# The 11-user deploy screen's PTQ price for the note's own int6 cut (it36int6split)
SCREEN_NOTE = (0.000268, 0.000307)


def load() -> dict:
    if not SWEEP.exists():
        raise SystemExit(f"no sweep yet: {SWEEP}")
    rows = json.loads(SWEEP.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        raise SystemExit("sweep has no rows")
    r = rows[0]
    for key in ("bits", "bits_ok", "gate"):
        if key not in r:
            raise SystemExit(f"sweep row is missing {key}")
    for head in ("ahead", "imm"):
        for key in ("delta", "ub95", "mean_cur", "mean_ref"):
            if key not in r["gate"].get(head, {}):
                raise SystemExit(f"sweep row is missing gate.{head}.{key}")
    return r


def landing():
    """The chain log's DONE line for the l2i6n eval - wall-clock landing time and duration."""
    if not CHAIN_LOG.exists():
        return "landed", 0.0
    txt = CHAIN_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] DONE  eval l2i6n@124: rc=0 in ([\d.]+) min", txt):
        pass
    if m is None:
        return "landed", 0.0
    return m.group(1)[:5], float(m.group(2))


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def build(r: dict):
    card, note = r["bits"]["card"], r["bits"]["note"]
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing()
    # the two streams' int6 costs, both measured DIRECTLY on 750 users
    note_cost = (a["delta"] - ROW17[0], i["delta"] - ROW17[1])
    card_cost = (ROW48[0] - ROW17[0], ROW48[1] - ROW17[1])
    implied = (ROW45[0] - ROW48[0], ROW45[1] - ROW48[1])   # row45 - row48 = the note's cut, same pairs
    ratio = note_cost[1] / card_cost[1] if card_cost[1] else float("inf")
    note_owns = ratio >= 3.0
    if note_owns:
        reading = (
            f"the NOTE owns the int6 tax: its 384 factor bits cost {ratio:.0f}x the card's 256 bits on the "
            f"immediate head, which agrees with the 11-user deploy screen (`it36int6split`, "
            f"{fmt(SCREEN_NOTE[0])}/{fmt(SCREEN_NOTE[1])} for the same cut) and SUPERSEDES row 48's "
            f"'proportional to the bits they save' reading, because the card's int6 is nearly free"
        )
    else:
        reading = (
            f"the int6 tax stays split between the streams (the note's 384 bits cost {ratio:.1f}x the "
            f"card's 256 on the immediate head), which agrees with row 48's reading and means the 11-user "
            f"screen's {fmt(SCREEN_NOTE[1])} for the note's cut was a PTQ price that QAT recovers"
        )
    desc = (
        "QAT card rank-2 int8 + int4 shifts (row 17's card), note rank-1 **int6** factors + int4 shifts "
        "- the NOTE half of chain 17's ladder, i.e. row 45's note left under row 17's card "
        f"({card:,}/{note:,} = {total:,} bits, {3584 - total} below row 17 and {total - 2944} above "
        f"row 45) (`l2i6n`, `it36int6card` point 2, landed {when}, {eval_min:.1f} min; speed cell = row "
        "17's measured `l2i8s4` path, because the factor width changes only the quantization LEVELS and "
        "the card stays rank-2 int8 with int4 shifts)"
    )
    if ok:
        verdict = (
            f"**PASS on both heads**: ahead sits at {a['ub95'] / CEIL:.2f}x of the ceiling (ub95 "
            f"{fmt(a['ub95'])}) and the immediate head at {i['ub95'] / CEIL:.2f}x (ub95 {fmt(i['ub95'])}). "
            f"Priced directly against row 17, the note's {3584 - total} int6 bits cost {fmt(note_cost[0])} "
            f"ahead / {fmt(note_cost[1])} imm and the card's 256 (row 48 minus row 17) cost "
            f"{fmt(card_cost[0])}/{fmt(card_cost[1])}, so {reading}"
        )
        cor = "pending"
    else:
        fa, fi = a["delta"] / CEIL, i["delta"] / CEIL
        bits_fail = ", ".join(w for w in (f"ahead {fa:.2f}x" if not pass_a else "",
                                         f"imm {fi:.2f}x" if not pass_i else "") if w)
        verdict = f"**FAIL**: {bits_fail} over the ceiling (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])})"
        cor = "-"
    row = (
        f"| 49 | {desc} | {card:,} ({card // 8}) | {note:,} ({note // 8}) | 2,290 | 0.800x | "
        f"{a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {fmt(a['delta'])} | {fmt(i['delta'])} | "
        f"{verdict} | {cor} |"
    )
    para = (
        f"\n\n**Row 49 (`l2i6n`) is the mirror image of row 48 and prices the NOTE's int6 cut directly at "
        f"3,200 bits.** Row 17's card with only the NOTE's factor width cut int8 -> int6 reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} against "
        f"`qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - "
        f"{'a PASS on both heads' if ok else 'a FAIL'}. Subtracting row 17 "
        f"({fmt(ROW17[0])}/{fmt(ROW17[1])}) gives the note's {3584 - total} factor bits directly: "
        f"{fmt(note_cost[0])} ahead / {fmt(note_cost[1])} imm. The card's 256 bits cost "
        f"{fmt(card_cost[0])}/{fmt(card_cost[1])} (row 48 minus row 17), and the row-45-minus-row-48 "
        f"difference implied {fmt(implied[0])}/{fmt(implied[1])} for the same note cut - "
        f"{'consistent with' if abs(note_cost[1] - implied[1]) <= 0.00015 else 'against'} this row's "
        f"direct measurement. Conclusion: {reading}."
    )
    note_bullet = (
        f"* 2026-11-23 03:10 - **Row 49 (`l2i6n`, {card:,}/{note:,} = {total:,} bits) - the NOTE half of "
        f"chain 17's ladder - {'PASSES both heads' if ok else 'FAILS'}.** Row 17's card with only the "
        f"note's factor width cut int8 -> int6 reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{fmt(a['delta'])}/{fmt(i['delta'])} vs `qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, "
        f"750 users). Direct pricing on 750 users: the note's {3584 - total} int6 bits cost "
        f"{fmt(note_cost[0])}/{fmt(note_cost[1])} against row 17, the card's 256 cost "
        f"{fmt(card_cost[0])}/{fmt(card_cost[1])} (row 48 minus row 17). So {reading}. "
        f"Finiteness: "
        f"{'the check is queued in `corruption/_watch_and_check_chain18.py` (the same queue hold), so the corruption cell stays pending' if ok else 'not needed - the row fails the loss gate'}.\n"
    )
    fin = ("Both rows' finiteness checks are queued in `corruption/_watch_and_check_chain18.py`."
           if ok else "This row fails the loss gate, so it needs no finiteness check.")
    board = (
        f"\n## Amendment 35 (2026-11-23 03:10): chain 18 point 2 (`l2i6n`) prices the NOTE's int6 cut "
        f"directly ({card:,}/{note:,} = {total:,} bits)\n\n"
        f"`it36int6card` point 2 holds the card at row 17's rank-2 int8 + int4 shifts and cuts only the "
        f"NOTE's factor width int8 -> int6: {card:,}/{note:,} = {total:,} bits, {3584 - total} below row "
        f"17 and {total - 2944} above row 45. It lands at {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{fmt(a['delta'])}/{fmt(i['delta'])} (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, "
        f"{mins:.1f} min) - {'PASS on both heads' if ok else '**FAIL**'}. With point 1 (`l2i6c`, card "
        f"int6 + note int8) the two rows give each stream's int6 cost directly on the full roster: the "
        f"note's {3584 - total} bits cost {fmt(note_cost[0])} ahead / {fmt(note_cost[1])} imm (row 49 "
        f"minus row 17) and the card's 256 bits cost {fmt(card_cost[0])}/{fmt(card_cost[1])} (row 48 "
        f"minus row 17), against the 11-user deploy screen's {fmt(SCREEN_NOTE[0])}/{fmt(SCREEN_NOTE[1])} "
        f"for the note's cut. Reading: {reading}. {fin}\n"
    )
    return row, para, note_bullet, board, ok


def edit(path: Path, fn):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
    lf = text.replace("\r\n", "\n")
    new = fn(lf)
    if new is None:
        return False
    out = new.replace("\n", "\r\n") if crlf else new
    path.write_bytes(out.encode("utf-8"))
    if path.read_bytes().decode("utf-8") != out:
        print(f"FAIL: {path.name} round trip differs")
        return False
    print(f"{path.name}: written ({len(raw)} -> {len(out.encode('utf-8'))} bytes, "
          f"{'CRLF' if crlf else 'LF'})")
    return True


def main() -> int:
    r = load()
    row, para, note_bullet, board, ok = build(r)
    print(f"verdict: {'PASS' if ok else 'FAIL'}  {' '.join(row.split())[:110]}...")

    para_anchor = ("which holds the queue so the checks run in the first free window after the chain's "
                   "end marker.")

    def do_results(lf: str):
        if "| 49 |" in lf:
            print("FAIL: row 49 already present")
            return None
        if lf.count(para_anchor) != 1:
            print(f"FAIL: paragraph anchor x{lf.count(para_anchor)}")
            return None
        lines = lf.split("\n")
        data = [n for n, l in enumerate(lines) if re.match(r"^\| \d+ \|", l)]
        if not data:
            print("FAIL: no data rows found")
            return None
        last = data[-1]
        lines.insert(last + 1, row)
        out = "\n".join(lines)
        j = out.index(para_anchor) + len(para_anchor)
        out = out[:j] + para + out[j:]
        got = [l for l in out.split("\n") if l.startswith("| 49 |")][0]
        if len(got.split("|")) != 14:
            print(f"FAIL: row 49 has {len(got.split('|')) - 2} cells, expected 12")
            return None
        print(f"  row 49 cells: {len(got.split('|')) - 2} (ok), inserted after data row {last}")
        return out

    def do_notes(lf: str):
        if "Row 49 (`l2i6n`" in lf:
            print("FAIL: NOTES bullet already present")
            return None
        anchor = "* 2026-11-23 02:45 - **The SHIFT-width knob is CLOSED on the immediate head"
        if lf.count(anchor) != 1:
            print(f"FAIL: NOTES anchor x{lf.count(anchor)}")
            return None
        return lf.replace(anchor, note_bullet + anchor, 1)

    def do_board(lf: str):
        if "Amendment 35" in lf:
            print("FAIL: Amendment 35 already present")
            return None
        if "Amendment 34" not in lf:
            print("FAIL: Amendment 34 missing - record row 48 first")
            return None
        return lf.rstrip("\n") + "\n" + board

    def do_recorded(lf: str):
        if f"{STAMP}\n" in lf:
            print("FAIL: stamp already present")
            return None
        return lf.rstrip("\n") + f"\n{STAMP}\n"

    okall = True
    okall = edit(RESULTS, do_results) and okall
    okall = edit(NOTES, do_notes) and okall
    okall = edit(BOARD, do_board) and okall
    okall = edit(RECORDED, do_recorded) and okall
    print("row 49 recorded" if okall else "SOMETHING FAILED")
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())
