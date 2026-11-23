"""Record chain 26's fixed-basis codec points (`it36fbrung`) as the next RESULTS rows.

WHAT THE POINTS ARE. Chain 19 (rows 50/51) showed the codec at m=8 and m=6 passes the loss gate, BEATS
the per-step rank-r factors at equal bits (+0.000040/+0.000129 vs +0.000052/+0.000327 at row 17's exact
3,584 bits), and at 2,912 bits beats row 45's own tax. The codec's payload is m*m*qbits for the whole
WKV instead of 2*K*r*qbits, so `m` is a size axis the uniform design does not have. Chain 26 walks it
DOWN, with row 17's card and int4 shifts held fixed so each point is a note-only A/B:
  * `l2i8fb4n` = note codec m=4 = 1,280/1,152 = **2,432 bits** - 512 BELOW row 45's 2,944-bit frontier,
    held-out capture 0.751-0.856 per layer;
  * `l2i8fb5n` = note codec m=5 = 1,280/1,368 = **2,648 bits** - 296 below that frontier, capture
    0.80-0.86.
The ladder is not gated point-on-point: if m=4 breaks the gate, m=5 is the rung that fills the gap.

ROW NUMBERS ARE DYNAMIC. Chain 26 can produce one row or two, and either may FAIL, so `next_row` reads
the table at run time and a point whose row already exists (recognised by its own config name plus the
`it36fbrung` stamp in the row's cells) is skipped instead of duplicated. Re-running after the second
point lands is therefore safe.

EVERY FIGURE COMES FROM THE SWEEP JSON at run time (`quant_sweep_it36<point>.json`): bits (now DERIVED
from the env by `quant_sweep.bits_for`, not the registry claim), both heads' deltas and ub95, the mean
losses and the wall clock. The speed cell comes from the codec's own engine probes; `l2i8fb4n` and
`l2i8fb5n` have never been timed, so until a `rust_speed.py` run names them the cell says UNMEASURED
and names the command that would settle it - it is NOT borrowed from m=6/m=8, because the m<16 slicing
is a different engine path.

AFTER THIS RUNS, the two corruption cells still need `corruption/record_corruption.py`, which resolves a
tag to a row only when exactly ONE row line names it - these rows name their tag in the narrative prose
too, so add explicit ROWS entries (the script prints the exact lines to add).

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_chain26.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]           # scratchpad/quant_plan/corruption -> repo root
RUN = ROOT / "scratchpad" / "quant_plan"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"
CHAIN_LOG = RUN / "it36fbrung_chain.log"
STAMP_MARK = "it36fbrung"

CEIL = 0.0005
SPEED_FLOOR = 0.75
ROW17 = (0.000052, 0.000327)   # 1,280/2,304 = 3,584 bits
ROW45 = (0.000125, 0.000381)   # 1,024/1,920 = 2,944 bits - the uniform-design frontier
ROW51 = (0.000018, 0.000209)   # 1,280/1,632 = 2,912 bits - the codec's own m=6 rung
FRONTIER = 2944

# (row id or None for dynamic, config name, sweep stamp, m, the codec's held-out capture range)
POINTS = [
    (None, "l2i8fb4n", "it36l2i8fb4n", 4, "0.751-0.856"),
    (None, "l2i8fb5n", "it36l2i8fb5n", 5, "0.80-0.86"),
]
# Speed JSONs to try, best first (the post-basis-layout-fix runs, then the older ones).
# "fbrung" (2026-11-23 09:47) is the run that actually NAMES the m=4 and m=5 rungs, with m=6 in
# the same session as a cross-check. It ran while chain 25 held the machine, so its absolute
# ratios are a CONSERVATIVE (low) reading of the codec path.
SPEED_TAGS = ["fbrung", "fb_speed3", "fb_speed2", "fb_idle", "fb_speed"]
LOADED_TAG = "fbrung"   # the tag measured beside a running chain, so its ratios read low


def speed_of(name):
    """(rev_s, x_rev, median per-user x, tag) for a config, from the first available speed JSON."""
    for tag in SPEED_TAGS:
        p = RUN / f"rust_speed_{tag}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if name not in d or "plain" not in d:
            continue
        r, b = d[name], d["plain"]
        common = sorted(set(r["per_user_rev_s"]) & set(b["per_user_rev_s"]))
        ratios = sorted(r["per_user_rev_s"][u] / b["per_user_rev_s"][u] for u in common)
        med = ratios[len(ratios) // 2] if ratios else float("nan")
        return r["rev_s"], r["rev_s"] / b["rev_s"], med, tag
    return None


def load(stamp: str):
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
    if not rows:
        return None
    r = rows[0]
    for key in ("bits", "bits_ok", "gate"):
        if key not in r:
            raise SystemExit(f"{stamp}: sweep row is missing {key}")
    for head in ("ahead", "imm"):
        for key in ("delta", "ub95", "mean_cur", "mean_ref"):
            if key not in r["gate"].get(head, {}):
                raise SystemExit(f"{stamp}: sweep row is missing gate.{head}.{key}")
    return r


def landing(name: str):
    """The chain log's DONE line for THIS point's eval - wall-clock landing time and duration."""
    if not CHAIN_LOG.exists():
        return "landed", 0.0
    txt = CHAIN_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(rf"\[(\d\d:\d\d:\d\d)\] DONE  eval {name}@124: rc=0 in ([\d.]+) min", txt):
        pass
    return (m.group(1)[:5], float(m.group(2))) if m else ("landed", 0.0)


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def cmp_word(x: float) -> str:
    """'+0.000052' is a COST of 0.000052, '-0.000012' is a WIN: never write a negative cost."""
    return f"BETTER by {abs(x):.6f}" if x < 0 else f"worse by {abs(x):.6f}"


def next_row(lf: str) -> int:
    ids = [int(n) for n in re.findall(r"^\| (\d+) \|", lf, re.M)]
    return (max(ids) + 1) if ids else 1


def already_recorded(lf: str, name: str) -> bool:
    """A row for this point exists when one data row names BOTH the config and this chain."""
    for line in lf.split("\n"):
        if re.match(r"^\| \d+ \|", line) and f"`{name}`" in line and STAMP_MARK in line:
            return True
    return False


def build(row_id, name, stamp, m, capture, r):
    card, note = r["bits"]["card"], r["bits"]["note"]
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing(name)
    vs17 = (a["delta"] - ROW17[0], i["delta"] - ROW17[1])
    vs45 = (a["delta"] - ROW45[0], i["delta"] - ROW45[1])
    vs51 = (a["delta"] - ROW51[0], i["delta"] - ROW51[1])
    below = FRONTIER - total

    sp = speed_of(name)
    if sp:
        rev, xrev, med, tag = sp
        sp_cells = f"{rev:,.1f} | {xrev:.3f}x"
        sp_ok = med >= SPEED_FLOOR
        sp_clause = (f"the deploy engine's codec path measures {rev:,.1f} rev/s = {xrev:.3f}x rev/s "
                     f"(median per-user {med:.3f}x) against `plain` in the same run "
                     f"(`rust_speed_{tag}.json`), so it "
                     f"{'CLEARS' if sp_ok else 'FAILS'} the 0.75x floor by "
                     f"{abs(med - SPEED_FLOOR):.3f}"
                     + (", measured while a chain job held the machine - the same probe reads "
                        "0.760x for the m=6 rung against 0.793x in `rust_speed_fb_speed3.json`, so "
                        "this figure is a conservative floor on the codec's speed"
                        if tag == LOADED_TAG else ""))
    else:
        rev = xrev = med = None
        sp_ok = None
        sp_cells = "unmeasured | unmeasured"
        sp_clause = (f"UNMEASURED: no `rust_speed_*.json` holds `{name}`, and the codec's m<16 slicing "
                     f"is its own engine path, so `rust_speed.py --configs plain,{name}` is what would "
                     f"settle the speed cell")
    near_floor = sp is not None and abs(med - SPEED_FLOOR) < 0.02

    desc = (
        f"QAT card rank-2 int8 + int4 shifts (row 17's card), note **FIXED-BASIS codec** m={m} int8 "
        f"({card:,}/{note:,} = {total:,} bits, {below} below the 2,944-bit frontier) - the note's state "
        f"stored as an m x m coefficient block against a basis SHIPPED with the model instead of the "
        f"rank-1 path's per-step factors (`{name}`, `{STAMP_MARK}` point {1 if m == 4 else 2}, landed "
        f"{when}, {eval_min:.1f} min; held-out capture {capture} at m={m}, against the shipped rank-1 "
        f"path's 0.904-0.951 per layer; speed is the codec path's own engine measurement, since speed "
        f"is per config and not per checkpoint)"
    )
    if ok:
        size_clause = (f"at {total:,} bits this is {below} BELOW row 45's frontier, so the frontier "
                       f"moves if it holds")
        verdict = (
            f"**PASS on both heads**: ahead at {a['ub95'] / CEIL:.2f}x of the ceiling (ub95 "
            f"{fmt(a['ub95'])}) and the immediate head at {i['ub95'] / CEIL:.2f}x (ub95 {fmt(i['ub95'])}). "
            f"Against row 17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}) the codec is "
            f"{cmp_word(vs17[0])}/{cmp_word(vs17[1])} for {3584 - total} fewer bits; against row 45 "
            f"({fmt(ROW45[0])}/{fmt(ROW45[1])}) {cmp_word(vs45[0])}/{cmp_word(vs45[1])}; against the "
            f"codec's own m=6 rung (row 51, {fmt(ROW51[0])}/{fmt(ROW51[1])}) "
            f"{cmp_word(vs51[0])}/{cmp_word(vs51[1])} - so this point is {size_clause}. Speed: {sp_clause}"
        )
    else:
        why = ", ".join(w for w in (f"ahead {a['delta'] / CEIL:.2f}x" if not pass_a else "",
                                    f"imm {i['delta'] / CEIL:.2f}x" if not pass_i else "") if w)
        verdict = (f"**FAIL**: {why} over the ceiling (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}); against "
                   f"row 17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}) the codec costs "
                   f"{fmt(vs17[0])}/{fmt(vs17[1])} and against row 45 "
                   f"({fmt(ROW45[0])}/{fmt(ROW45[1])}) {fmt(vs45[0])}/{fmt(vs45[1])}, so this rung of "
                   f"the ladder is closed at {total:,} bits. Speed: {sp_clause}")
    row = (
        f"| {row_id} | {desc} | {card:,} ({card // 8}) | {note:,} ({note // 8}) | {sp_cells} | "
        f"{a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {fmt(a['delta'])} | {fmt(i['delta'])} "
        f"| {verdict} | {'pending' if ok else '-'} |"
    )
    para = (
        f"\n\n**Row {row_id} (`{name}`) is the fixed-basis codec at m={m} ({card:,}/{note:,} = "
        f"{total:,} bits): the codec's own size ladder, {below} below row 45's frontier.** "
        f"The note's state is an m x m coefficient block against a basis fitted on 3 users of the row-17 "
        f"checkpoint and scored LEAVE-ONE-USER-OUT; held out it captures {capture} per layer, against "
        f"the shipped rank-1 note's 0.904-0.951 - so on capture alone this rung looks worse than every "
        f"rank-r row, which is exactly why the loss has to be measured. It reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} against "
        f"`qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - "
        f"{'a PASS on both heads' if ok else 'a FAIL'}. Against row 17 "
        f"({fmt(ROW17[0])}/{fmt(ROW17[1])}, which costs {3584 - total} MORE bits) the codec is "
        f"{cmp_word(vs17[0])}/{cmp_word(vs17[1])}; against row 45 ({fmt(ROW45[0])}/{fmt(ROW45[1])}) "
        f"{cmp_word(vs45[0])}/{cmp_word(vs45[1])}; and against the codec's own m=6 rung (row 51, "
        f"{fmt(ROW51[0])}/{fmt(ROW51[1])}) {cmp_word(vs51[0])}/{cmp_word(vs51[1])}. "
        f"{'So the codec keeps buying size at no loss cost, and the frontier moves.' if ok else 'So this rung closes the ladder at this size.'} "
        f"Speed: {sp_clause}"
        f"{', a margin inside 2% of the floor, so an idle re-measure is the number to quote' if near_floor else ''}."
    )
    if ok and sp_ok is False:
        para += (" The row PASSES the loss gate but FAILS the speed floor, so it is NOT deployable as "
                 "measured.")
    note_bullet = (
        f"* {time.strftime('%Y-%m-%d %H:%M')} - **Chain 26 point (m={m}) `{name}` lands as RESULTS row "
        f"{row_id}: {card:,}/{note:,} = {total:,} bits, {'PASS' if ok else 'FAIL'}.** "
        f"The fixed-basis codec at m={m} (held-out capture {capture} against the shipped rank-1 note's "
        f"0.904-0.951) reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/"
        f"{fmt(i['delta'])} vs `qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, "
        f"{mins:.1f} min). Against row 17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}, {3584 - total} more bits) "
        f"{cmp_word(vs17[0])}/{cmp_word(vs17[1])}; against row 45 "
        f"({fmt(ROW45[0])}/{fmt(ROW45[1])}) {cmp_word(vs45[0])}/{cmp_word(vs45[1])}; against row 51 "
        f"({fmt(ROW51[0])}/{fmt(ROW51[1])}) {cmp_word(vs51[0])}/{cmp_word(vs51[1])}. "
        f"{'That is 512+ bits below the 2,944 frontier the uniform design reached.' if below >= 296 else ''}"
        f"{'The ladder closes here: the codec pays too much at this size.' if not ok else ''}"
        f" Speed: {sp_clause}.\n"
    )
    fin = ("a corruption check runs in this chain's own phase"
           if ok else "not needed - the row fails the loss gate")
    board = (
        f"\n## Amendment {AMEND[0]} (2026-11-23 {time.strftime('%H:%M')}): chain 26 point "
        f"(m={m}) `{name}` - fixed-basis codec, {card:,}/{note:,} = {total:,} bits\n\n"
        f"The codec's own size ladder (chain 19 measured m=8 and m=6; this is m={m}). `{name}` holds "
        f"row 17's card at rank-2 int8 + int4 shifts and the note as an m x m coefficient block against "
        f"the shipped basis: {card:,}/{note:,} = {total:,} bits, {below} below row 45's 2,944-bit "
        f"frontier. Landed {when} after {eval_min:.1f} min of eval. It reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} (ub95 "
        f"{fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users) - {'PASS on both heads' if ok else '**FAIL**'}. "
        f"Against row 17 ({3584 - total} more bits) the codec is "
        f"{cmp_word(vs17[0])}/{cmp_word(vs17[1])}; against row 45 {cmp_word(vs45[0])}/"
        f"{cmp_word(vs45[1])}; against the m=6 rung (row 51) {cmp_word(vs51[0])}/"
        f"{cmp_word(vs51[1])}. Its held-out capture is {capture} at m={m} against the rank-1 note's "
        f"0.904-0.951, so capture still does not predict the loss. Speed: {sp_clause}. "
        f"Finiteness: {fin}.\n"
    )
    return row, para, note_bullet, board, ok, sp_ok, med


AMEND = [40]


def next_amendment(lf: str) -> int:
    nums = [int(n) for n in re.findall(r"^## Amendment (\d+)", lf, re.M)]
    return max(nums) + 1 if nums else 1


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
    AMEND[0] = next_amendment(BOARD.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"))
    print(f"next board amendment: {AMEND[0]}")
    rl = RESULTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    row_id = next_row(rl)
    todo = []
    for _row, name, stamp, m, capture in POINTS:
        if already_recorded(rl, name):
            print(f"{name}: already recorded - skipping")
            continue
        r = load(stamp)
        if r is None:
            print(f"{name}: no sweep yet ({stamp})")
            continue
        todo.append((row_id, name, stamp, m, capture, build(row_id, name, stamp, m, capture, r)))
        row_id += 1
        AMEND[0] += 1
    if not todo:
        print("nothing to record")
        return 0

    okall = True
    for row_id, name, stamp, m, capture, (row, para, note_bullet, board, ok, sp_ok, med) in todo:
        print(f"verdict row {row_id} ({name}, m={m}): {'PASS' if ok else 'FAIL'} "
              f"{' '.join(row.split())[:100]}...")
        if ok and sp_ok is None:
            print(f"  WARNING: row {row_id} passes the loss gate but its SPEED is unmeasured - it needs "
                  f"`rust_speed.py --configs plain,{name}` before it is deployable")
        elif ok and sp_ok is False:
            print(f"  WARNING: row {row_id} PASSES the loss gate but its speed is {med:.3f}x median "
                  f"per-user, BELOW the 0.75x floor - it is not deployable as measured")
        elif ok and med is not None and abs(med - SPEED_FLOOR) < 0.02:
            print(f"  NOTE: row {row_id} clears the speed floor by only {abs(med - SPEED_FLOOR):.3f} "
                  f"({med:.3f}x) - an idle re-measure is the number to quote")

        def do_results(lf: str, row_id=row_id, row=row, para=para):
            lines = lf.split("\n")
            data = [n for n, l in enumerate(lines) if re.match(r"^\| \d+ \|", l)]
            if not data:
                print("FAIL: no data rows found")
                return None
            lines.insert(data[-1] + 1, row)
            out = "\n".join(lines)
            anchor = "which holds the queue so the checks run in the first free window after the chain's end marker."
            if out.count(anchor) == 1:
                j = out.index(anchor) + len(anchor)
                out = out[:j] + para + out[j:]
            else:
                print("  note: paragraph anchor not unique - paragraph appended at the table's end")
                out = out.rstrip("\n") + para + "\n"
            got = [l for l in out.split("\n") if re.match(rf"^\|\s*{row_id}\s*\|", l)]
            if len(got) != 1:
                print(f"FAIL: row {row_id} appears {len(got)} times")
                return None
            if len(got[0].split("|")) != 14:
                print(f"FAIL: row {row_id} has {len(got[0].split('|')) - 2} cells, expected 12")
                return None
            print(f"  row {row_id} cells: 12 (ok)")
            return out

        def do_notes(lf: str, name=name, bullet=note_bullet):
            if f"`{name}` lands as RESULTS row" in lf:
                print("FAIL: NOTES bullet already present")
                return None
            m2 = re.search(r"^\* \d{4}-\d\d-\d\d \d\d:\d\d - ", lf, re.M)
            if not m2:
                print("FAIL: no NOTES bullet anchor found")
                return None
            return lf[:m2.start()] + bullet + lf[m2.start():]

        def do_board(lf: str, board=board):
            mk = re.search(r": chain 26 point \(m=(\d+)\) `([^`]*)`", board)
            key = mk.group(2) if mk else None
            if key and re.search(rf": chain 26 point \(m=\d+\) `{re.escape(key)}`", lf):
                print(f"skip: the BOARD already carries a chain-26 amendment for {key}")
                return lf
            n = next_amendment(lf)
            body = re.sub(r"^## Amendment \d+", f"## Amendment {n}", board.lstrip("\n"), count=1)
            return lf.rstrip("\n") + "\n" + body

        def do_recorded(lf: str, stamp=stamp):
            if f"{stamp}\n" in lf:
                print("FAIL: stamp already present")
                return None
            return lf.rstrip("\n") + f"\n{stamp}\n"

        okall = edit(RESULTS, do_results) and okall
        okall = edit(NOTES, do_notes) and okall
        okall = edit(BOARD, do_board) and okall
        okall = edit(RECORDED, do_recorded) and okall

    print("\nNEXT STEP: fill the corruption cells with `corruption/record_corruption.py`; these rows")
    print("name their tag in the prose too, so add explicit ROWS entries first, e.g.:")
    for _row, name, stamp, m, capture in POINTS:
        print(f'    "{name}": "<row id>",    # chain 26 point (codec note m={m})')
    print("chain 26 points recorded" if okall else "SOMETHING FAILED")
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())