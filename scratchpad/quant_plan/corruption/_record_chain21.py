"""Record chain 21's points (`it36int6combo`) as the next RESULTS rows.

WHAT THE POINTS ARE. Chain 21 stacks the int6 CARD rung (price: +0.000052 ahead / +0.000115 imm on 11
long histories, `rust_wide_screen_it36int6split.json`) on the note's int3 SHIFT rung (price: +0.000011
ahead / +0.000037 imm on 750 users, row 35):
  * point 1 `l2i6s4n1i8s3` = card rank-2 int6 + int4 shifts (1,024) + note rank-1 int8 + **int3**
    shifts (2,112) = **3,136 bits** (always runs). Against row 35 (3,392: card rank-2 int8 + the same
    note int3 shifts) it prices the card's int8 -> int6 cut; against row 45 (2,944) it prices the note's
    int4 -> int3 shift cut, bought for +192 note bits.
  * point 2 `l2i6s4n1i8s2` = the same card + note rank-1 int8 + **int2** shifts (1,920) = **2,944 bits**,
    EXACTLY row 45's size bought the other way (int8 factors + int2 shifts vs int6 factors + int4
    shifts), so it is a clean A/B at equal bits. It runs only if point 1's own sweep PASSES the
    log-loss gate (the chain's own gate), and its corruption check runs only if its own sweep passes.

ROW NUMBERS ARE DYNAMIC on purpose: chain 21 may produce one row or two, and an earlier landing that
cannot be recorded (a missing sweep JSON) must not shift the numbering. `next_row` reads the table at
run time; a row id already present is skipped, so re-running after the second point lands is safe.

SPEED CELLS. RESULTS.md quotes speed PER CONFIG, so each point looks for its own config in every
`rust_speed_*.json` and falls back to a named path-equivalent:
  * point 1 falls back to `n1i8s3` (row 35's own engine probe, `rust_speed_underload_n1i8s3.json`,
    users 7213/7250 while a roster sweep held the GPU: plain 2,732.5 rev/s, `n1i8s3` 2,155.5 = 0.789x
    rev, median per-user 0.791). The card's factor width does not change the card's verify path, so that
    IS this point's path.
  * point 2's note int2 shift path has never been measured: the per-stream shift map is new (row 35's
    finding) and no probe sets `note:int2`. It is recorded as unmeasured, with the run that would settle
    it named, rather than borrowed from the CARD's int2 number (row 34's `l2i8s2`), which is a different
    stream.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_chain21.py
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
CHAIN_LOG = RUN / "it36int6combo_chain.log"

CEIL = 0.0005
SPEED_FLOOR = 0.75

# (candidate, sweep stamp, card bits, note bits, speed alias or None, why the alias IS this path)
POINTS = [
    ("l2i6s4n1i8s3", "it36l2i6s4n1i8s3", 1024, 2112, "n1i8s3",
     "row 35's own note path: the card's factor width does not change its verify path"),
    ("l2i6s4n1i8s2", "it36l2i6s4n1i8s2", 1024, 1920, None, None),
]
REF_ROW35 = (0.000011, 0.000037, "3,392 bits - card rank-2 int8 + the same note int3 shifts")
REF_ROW45 = (0.000125, 0.000381, "2,944 bits - card rank-2 int6 + note rank-1 int6 + int4 shifts")
REF_ROW17 = (0.000052, 0.000327, "3,584 bits - card rank-2 int8 + note rank-1 int8 + int4 shifts")

# tag preference when the same config appears in several probes (an idle-ish probe first)
SPEED_PREF = ["idle_l2i8", "underload_n1i8s3", "rank2card", "fb_idle", "fb_speed", "fastrank_on",
              "fastrank_off", "fastrow17", "bundle", "probe_note2"]


def speed_files():
    out = []
    for tag in SPEED_PREF:
        p = RUN / f"rust_speed_{tag}.json"
        if p.exists():
            out.append(p)
    for p in sorted(RUN.glob("rust_speed_*.json")):
        if p not in out:
            out.append(p)
    return out


def speed_of(names):
    """(name used, rev_s, x_rev, median per-user x, tag) for the first name found in any speed JSON."""
    if isinstance(names, str):
        names = [names]
    for p in speed_files():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if "plain" not in d:
            continue
        for n in names:
            if n not in d:
                continue
            r, b = d[n], d["plain"]
            common = sorted(set(r["per_user_rev_s"]) & set(b["per_user_rev_s"]))
            ratios = sorted(r["per_user_rev_s"][u] / b["per_user_rev_s"][u] for u in common)
            med = ratios[len(ratios) // 2] if ratios else float("nan")
            return n, r["rev_s"], r["rev_s"] / b["rev_s"], med, p.stem.replace("rust_speed_", "")
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
    if not CHAIN_LOG.exists():
        return "landed", 0.0
    txt = CHAIN_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(rf"\[(\d\d:\d\d:\d\d)\] DONE  eval {name}@124: rc=0 in ([\d.]+) min", txt):
        pass
    return (m.group(1)[:5], float(m.group(2))) if m else ("landed", 0.0)


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def next_row(lf: str) -> int:
    ids = [int(n) for n in re.findall(r"^\| (\d+) \|", lf, re.M)]
    return (max(ids) + 1) if ids else 1


def build(row_id, name, stamp, card, note, speed_alias, alias_why, r):
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing(name)
    note_lvl = 3 if name.endswith("s3") else 2
    card_f = "int6" if "i6" in name else "int8"

    sp = speed_of([name] + ([speed_alias] if speed_alias else []))
    if sp:
        used, rev, xrev, med, tag = sp
        sp_cells = f"{rev:,.1f} | {xrev:.3f}x"
        sp_ok = med >= SPEED_FLOOR
        alias_clause = ("" if used == name else
                        f"; this is the `{used}` path's own probe, and {alias_why}")
        sp_clause = (f"{rev:,.1f} rev/s = {xrev:.3f}x rev/s, median per-user {med:.3f}x, against "
                     f"`plain` in the same run (`rust_speed_{tag}.json`{alias_clause}), so it "
                     f"{'CLEARS' if sp_ok else 'FAILS'} the 0.75x floor")
    else:
        rev = xrev = med = None
        sp_ok = None
        sp_cells = "unmeasured | unmeasured"
        sp_clause = ("UNMEASURED: no `rust_speed_*.json` holds this config, and the per-stream shift "
                     "map is new, so a `rust_speed.py` run with "
                     "`RWKV_STATE_SHIFT_LEVEL=card:int4,note:int%d` is what would settle it" % note_lvl)
    near_floor = sp is not None and abs(med - SPEED_FLOOR) < 0.02

    desc = (
        f"QAT card rank-2 **{card_f}** factors + int4 shifts ({card:,} bits), note rank-1 **int8** "
        f"factors + **int{note_lvl}** shifts ({note:,} bits) - chain 21's stacked rung "
        f"(`{name}`, `it36int6combo`, landed {when}, {eval_min:.1f} min). {card + note:,} bits = "
        f"row 35's {REF_ROW35[2]} minus the card's 256 int6 bits, and "
        f"{'EXACTLY row 45s size bought the other way (int8 factors + int2 shifts vs int6 factors + '
           'int4 shifts), so it is a clean A/B' if total == 2944 else f'{total - 2944} bits ABOVE row 45s 2,944 frontier'}; "
        f"speed quoted per config ({sp_cells.replace(' | ', ' / ')})"
    )
    vs35 = (a["delta"] - REF_ROW35[0], i["delta"] - REF_ROW35[1])
    vs45 = (a["delta"] - REF_ROW45[0], i["delta"] - REF_ROW45[1])
    if ok:
        verdict = (
            f"**PASS on both heads**: ahead at {a['ub95'] / CEIL:.2f}x of the ceiling (ub95 "
            f"{fmt(a['ub95'])}) and the immediate head at {i['ub95'] / CEIL:.2f}x (ub95 {fmt(i['ub95'])}). "
            f"Against row 35 ({fmt(REF_ROW35[0])}/{fmt(REF_ROW35[1])}, {REF_ROW35[2]}) it costs "
            f"{fmt(vs35[0])}/{fmt(vs35[1])}; against row 45 ({fmt(REF_ROW45[0])}/{fmt(REF_ROW45[1])}) "
            f"{fmt(vs45[0])}/{fmt(vs45[1])}. Its speed: {sp_clause}"
        )
    else:
        fail = ", ".join(w for w in (f"ahead {a['delta'] / CEIL:.2f}x" if not pass_a else "",
                                     f"imm {i['delta'] / CEIL:.2f}x" if not pass_i else "") if w)
        verdict = (f"**FAIL**: {fail} over the ceiling (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}); "
                   f"against row 35 ({fmt(REF_ROW35[0])}/{fmt(REF_ROW35[1])}) it costs "
                   f"{fmt(vs35[0])}/{fmt(vs35[1])} and against row 45 "
                   f"({fmt(REF_ROW45[0])}/{fmt(REF_ROW45[1])}) {fmt(vs45[0])}/{fmt(vs45[1])}. "
                   f"Speed: {sp_clause}")
    row = (
        f"| {row_id} | {desc} | {card:,} ({card // 8}) | {note:,} ({note // 8}) | {sp_cells} | "
        f"{a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {fmt(a['delta'])} | {fmt(i['delta'])} "
        f"| {verdict} | {'pending' if ok else '-'} |"
    )
    if total == 2944:
        what = ("At row 45's exact size this is the direct A/B of the two ways to buy 2,944 bits: int8 "
                "note factors with int2 shifts, or int6 factors with int4 shifts. The answer here is "
                f"{'that the int8-factor form is the better buy' if vs45[1] < 0 else 'that the int6-factor form stays the better buy'}"
                f" ({fmt(vs45[1])} on the immediate head against row 45; a difference of deltas, not a "
                f"paired test).")
        short = ("This is row 45's exact size bought the other way, so the A/B says the "
                 f"{'int8-note-factor form wins' if vs45[1] < 0 else 'int6-factor form stays better'}.")
    else:
        what = (f"At {total:,} bits this is a diagnostic that isolates the two cut prices rather than a "
                f"size claim: the current frontier is row 45 at 2,944.")
        short = "This row is a diagnostic, not a new frontier."
    para = (
        f"\n\n**Row {row_id} (`{name}`) is chain 21's stacked rung ({card:,}/{note:,} = {total:,} "
        f"bits): the {card_f} CARD on the note's int{note_lvl} shift rung.** "
        f"It reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} against "
        f"`qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - "
        f"{'a PASS on both heads' if ok else 'a FAIL'}. "
        f"Priced against row 35 ({fmt(REF_ROW35[0])}/{fmt(REF_ROW35[1])}, {REF_ROW35[2]}) the card's "
        f"int8 -> int6 cut plus the note's shift cut together cost {fmt(vs35[0])}/{fmt(vs35[1])}; priced "
        f"against row 45 ({fmt(REF_ROW45[0])}/{fmt(REF_ROW45[1])}, {REF_ROW45[2]}) this layout costs "
        f"{fmt(vs45[0])}/{fmt(vs45[1])}. {what} Its speed: {sp_clause}"
        f"{', a margin inside 2% of the floor, so an idle re-measure is the number to quote' if near_floor else ''}."
    )
    note_bullet = (
        f"* {time.strftime('%Y-%m-%d %H:%M')} - **Chain 21 point `{name}` lands as RESULTS row {row_id}: "
        f"{card:,}/{note:,} = {total:,} bits, {'PASS' if ok else 'FAIL'}.** "
        f"Card rank-2 {card_f} + int4 shifts with the note at rank-1 int8 + int{note_lvl} shifts reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} vs "
        f"`qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min). Against row "
        f"35 ({fmt(REF_ROW35[0])}/{fmt(REF_ROW35[1])}) {fmt(vs35[0])}/{fmt(vs35[1])}; against row 45 "
        f"({fmt(REF_ROW45[0])}/{fmt(REF_ROW45[1])}) {fmt(vs45[0])}/{fmt(vs45[1])}. {short} "
        f"Speed: {sp_clause}.\n"
    )
    fin = ("a corruption check runs in this chain's own phase"
           if ok else "not needed - the row fails the loss gate")
    board = (
        f"\n## Amendment {AMEND[0]} (2026-11-23 {time.strftime('%H:%M')}): chain 21 point `{name}` - the "
        f"int6 card on the note's int{note_lvl} shift rung, {card:,}/{note:,} = {total:,} bits\n\n"
        f"`{name}` holds the int6 card (rank-2 {card_f} + int4 shifts, {card:,} bits) and the note at "
        f"rank-1 int8 + int{note_lvl} shifts ({note:,} bits): {total:,} bits, "
        f"{'exactly row 45s size' if total == 2944 else f'{total - 2944} above row 45'}. Landed {when} "
        f"after {eval_min:.1f} min. It reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{fmt(a['delta'])}/{fmt(i['delta'])} (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users) - "
        f"{'PASS on both heads' if ok else '**FAIL**'}. Priced against row 35 it costs "
        f"{fmt(vs35[0])}/{fmt(vs35[1])}; against row 45 {fmt(vs45[0])}/{fmt(vs45[1])}. Speed: "
        f"{sp_clause}. The row's finiteness {fin}.\n"
    )
    return row, para, note_bullet, board, ok, sp_ok, med


AMEND = [38]


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
    todo = []
    row_id = next_row(RESULTS.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"))
    print(f"next RESULTS row id: {row_id}")
    for name, stamp, card, note, alias, why in POINTS:
        r = load(stamp)
        if r is None:
            print(f"{name}: no sweep yet ({stamp})")
            continue
        if r["bits"]["card"] != card or r["bits"]["note"] != note:
            print(f"  WARNING: {name} sweep says {r['bits']['card']}/{r['bits']['note']}, "
                  f"the script expects {card}/{note}")
        todo.append((row_id, name, stamp,
                     build(row_id, name, stamp, card, note, alias, why, r)))
        row_id += 1
        AMEND[0] += 1
    if not todo:
        print("nothing to record")
        return 0

    okall = True
    for row_id, name, stamp, built in todo:
        row, para, note_bullet, board, ok, sp_ok, med = built
        print(f"verdict row {row_id}: {'PASS' if ok else 'FAIL'} {' '.join(row.split())[:90]}...")
        if ok and sp_ok is False:
            print(f"  WARNING: row {row_id} PASSES the loss gate but its speed is {med:.3f}x median "
                  f"per-user, BELOW the 0.75x floor - not deployable as measured")
        elif ok and sp_ok is None:
            print(f"  WARNING: row {row_id} passes the loss gate but its SPEED is unmeasured")

        def do_results(lf: str, row_id=row_id, row=row, para=para):
            if re.search(rf"^\|\s*{row_id}\s*\|", lf, re.M):
                print(f"FAIL: row {row_id} already present")
                return None
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
            got = [l for l in out.split("\n") if re.match(rf"^\|\s*{row_id}\s*\|", l)][0]
            if len(got.split("|")) != 14:
                print(f"FAIL: row {row_id} has {len(got.split('|')) - 2} cells, expected 12")
                return None
            print(f"  row {row_id} cells: {len(got.split('|')) - 2} (ok)")
            return out

        def do_notes(lf: str, row_id=row_id, bullet=note_bullet):
            if f"lands as RESULTS row {row_id}" in lf:
                print("FAIL: NOTES bullet already present")
                return None
            m = re.search(r"^\* \d{4}-\d\d-\d\d \d\d:\d\d - ", lf, re.M)
            if not m:
                print("FAIL: no NOTES bullet anchor found")
                return None
            return lf[:m.start()] + bullet + lf[m.start():]

        def do_board(lf: str, board=board):
            return lf.rstrip("\n") + "\n" + board

        def do_recorded(lf: str, stamp=stamp):
            if f"{stamp}\n" in lf:
                print("FAIL: stamp already present")
                return None
            return lf.rstrip("\n") + f"\n{stamp}\n"

        okall = edit(RESULTS, do_results) and okall
        okall = edit(NOTES, do_notes) and okall
        okall = edit(BOARD, do_board) and okall
        okall = edit(RECORDED, do_recorded) and okall
    print("chain 21 points recorded" if okall else "SOMETHING FAILED")
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())
