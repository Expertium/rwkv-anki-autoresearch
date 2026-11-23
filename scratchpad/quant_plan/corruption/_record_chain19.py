"""Record chain 19's fixed-basis points as RESULTS rows 50 (`l2i8fb8n`) and 51 (`l2i8fb6n`).

WHAT THE TWO POINTS ARE (chain 19 = `it36fixedbasis`, Andrew approved approach 1 on 2026-11-22 20:51).
The codec stores the m x m coefficient block C = P^T W Q against a FIXED basis that ships with the model,
instead of the rank-r path's per-step 2*K*r factors. The basis was fitted on 3 users of the row-17
checkpoint and scored LEAVE-ONE-USER-OUT (`corruption/fixedbasis_l2i8s4all.json`): held out it captures
0.876-0.921 at m=8 and 0.82-0.87 at m=6, against the note's shipped rank-1 capture of 0.904-0.951 per
layer. The two points price that:
  * `l2i8fb8n` = card rank-2 int8 + fixed-basis note m=8 = 1,280/2,304 = 3,584 bits = row 17's EXACT
    size, so it is a pure codec A/B: same bits, does the codec match the per-step factors?
  * `l2i8fb6n` = card rank-2 int8 + fixed-basis note m=6 = 1,280/1,632 = 2,912 bits = 672 below row 17
    and 32 BELOW row 45 - the first claim under the 2,944-bit frontier the uniform design space reached.

WHY IT IS WRITTEN BEFORE EITHER SWEEP LANDS: Andrew's standing order is to record and push the moment a
sweep lands, and either point may PASS or FAIL. Every number and verdict word is therefore read from
`quant_sweep_it36<point>.json` at run time, and the script records whichever point has landed (running it
again later records the other one). A row whose id is already present is skipped.

SPEED CELL: left as "unmeasured" on purpose. The codec changes `compress_wkv_state`, so row 17's measured
2,290 rev/s = 0.800x does NOT transfer the way it does for a factor-width change. The script prints a
warning when a point passes the loss gate, because such a row still needs its own `rust_speed.py` run
before it can be called deployable.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_chain19.py
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
CHAIN_LOG = RUN / "it36fixedbasis_chain.log"

CEIL = 0.0005
ROW17 = (0.000052, 0.000327)   # 1,280/2,304 = 3,584 bits, both streams rank-1/2 int8 + int4 shifts
ROW45 = (0.000125, 0.000381)   # 1,024/1,920 = 2,944 bits, both factor widths at int6

# (row id, config name, sweep stamp, the codec's held-out capture for that m, what the row compares to)
POINTS = [
    (50, "l2i8fb8n", "it36l2i8fb8n", "0.876-0.921", "row 17 (the SAME 3,584 bits)"),
    (51, "l2i8fb6n", "it36l2i8fb6n", "0.82-0.87", "row 45's 2,944-bit frontier, 32 bits above it"),
]


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


def landing(stamp: str):
    """The chain log's DONE line for the point's eval - wall-clock landing time and duration."""
    if not CHAIN_LOG.exists():
        return "landed", 0.0
    txt = CHAIN_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(rf"\[(\d\d:\d\d:\d\d)\] DONE  eval \w+@124: rc=0 in ([\d.]+) min", txt):
        pass
    return (m.group(1)[:5], float(m.group(2))) if m else ("landed", 0.0)


def fmt(x: float) -> str:
    return f"{x:+.6f}"


def build(row_id, name, stamp, capture, compare, r):
    card, note = r["bits"]["card"], r["bits"]["note"]
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing(stamp)
    vs17 = (a["delta"] - ROW17[0], i["delta"] - ROW17[1])
    vs45 = (a["delta"] - ROW45[0], i["delta"] - ROW45[1])
    which = "fb8n" if name.endswith("fb8n") else "fb6n"
    m = 8 if which == "fb8n" else 6
    desc = (
        f"QAT card rank-2 int8 + int4 shifts (row 17's card), note **FIXED-BASIS codec** m={m} int8 "
        f"({card:,}/{note:,} = {total:,} bits, {3584 - total} below row 17 and {total - 2944} "
        f"{'above' if total > 2944 else 'below'} row 45) - the note's state stored as an m x m "
        f"coefficient block against a basis SHIPPED with the model instead of the rank-1 path's per-step "
        f"factors (`{name}`, `it36fixedbasis` point {1 if which == 'fb8n' else 2}, landed {when}, "
        f"{eval_min:.1f} min; the codec's held-out capture is {capture} at m={m}, against the shipped "
        f"rank-1 path's 0.904-0.951 per layer; speed cell UNMEASURED because the codec changes "
        f"`compress_wkv_state`, so row 17's 0.800x does not transfer)"
    )
    if ok:
        size_clause = ("at row 17's EXACT size, so the codec's lower energy capture costs "
                       f"{fmt(vs17[0])}/{fmt(vs17[1])} for the SAME bits"
                       if total == 3584 else
                       f"with {3584 - total} fewer bits than row 17, so the codec's lower energy capture "
                       f"is bought back by the size cut")
        verdict = (
            f"**PASS on both heads**: ahead at {a['ub95'] / CEIL:.2f}x of the ceiling (ub95 "
            f"{fmt(a['ub95'])}) and the immediate head at {i['ub95'] / CEIL:.2f}x (ub95 {fmt(i['ub95'])}). "
            f"Against row 17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}) it costs {fmt(vs17[0])}/{fmt(vs17[1])}, "
            f"{size_clause}"
        )
    else:
        bits_fail = ", ".join(w for w in (f"ahead {a['delta'] / CEIL:.2f}x" if not pass_a else "",
                                         f"imm {i['delta'] / CEIL:.2f}x" if not pass_i else "") if w)
        verdict = (f"**FAIL**: {bits_fail} over the ceiling (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}); "
                   f"against row 17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}) the codec costs "
                   f"{fmt(vs17[0])}/{fmt(vs17[1])}")
    row = (
        f"| {row_id} | {desc} | {card:,} ({card // 8}) | {note:,} ({note // 8}) | unmeasured | "
        f"unmeasured | {a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {fmt(a['delta'])} | {fmt(i['delta'])} "
        f"| {verdict} | {'pending' if ok else '-'} |"
    )
    head = f"**Row {row_id} (`{name}`)"
    para = (
        f"\n\n{head} is chain 19's fixed-basis codec at m={m} ({card:,}/{note:,} = {total:,} bits).** "
        f"The codec stores an m x m coefficient block against a basis fitted on 3 users of the row-17 "
        f"checkpoint and scored LEAVE-ONE-USER-OUT; held out it captures {capture}, against the shipped "
        f"rank-1 note's 0.904-0.951 per layer. It reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{fmt(a['delta'])}/{fmt(i['delta'])} against `qnj_it36ctl` (ub95 {fmt(a['ub95'])}/"
        f"{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - "
        f"{'a PASS on both heads' if ok else 'a FAIL'}. Priced against row 17 "
        f"({fmt(ROW17[0])}/{fmt(ROW17[1])}), the codec costs {fmt(vs17[0])}/{fmt(vs17[1])}; priced "
        f"against row 45 ({fmt(ROW45[0])}/{fmt(ROW45[1])}), {fmt(vs45[0])}/{fmt(vs45[1])}. "
        f"{'So a FIXED basis of this size is competitive with the per-step rank-r factors' if ok else 'So the fixed basis pays more than the per-step factors do at this size'}"
        f". The speed cell is deliberately unmeasured: the codec replaces `compress_wkv_state` in the "
        f"deploy engine, so row 17's 0.800x does not transfer and this row needs its own "
        f"`rust_speed.py` run before it can be called deployable."
    )
    note_bullet = (
        f"* {time.strftime('%Y-%m-%d %H:%M')} - **Chain 19 point ({which}) `{name}` lands as RESULTS row "
        f"{row_id}: {card:,}/{note:,} = {total:,} bits, {'PASS' if ok else 'FAIL'}.** "
        f"The fixed-basis codec (m={m}, basis fitted on 3 users of the row-17 checkpoint, held-out "
        f"capture {capture} against the shipped rank-1 note's 0.904-0.951) reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} vs "
        f"`qnj_it36ctl` (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min). Against row "
        f"17 ({fmt(ROW17[0])}/{fmt(ROW17[1])}) it costs {fmt(vs17[0])}/{fmt(vs17[1])}; against row 45 "
        f"({fmt(ROW45[0])}/{fmt(ROW45[1])}) {fmt(vs45[0])}/{fmt(vs45[1])}. "
        f"{'At 3,584 bits this is row 17s exact size, so the codec MATCHES the per-step factors for the same bits.' if which == 'fb8n' and ok else ''}"
        f"{'At 2,912 bits this is 32 bits BELOW row 45s frontier, so the frontier has moved.' if which == 'fb6n' and ok else ''}"
        f"{'The codec loses to the per-step factors at this size.' if not ok else ''}"
        f"{' Its speed cell is UNMEASURED (the codec changes `compress_wkv_state`), so a `rust_speed.py` run is still needed before it is deployable.' if ok else ''}\n"
    )
    fin = ("a corruption check runs in this chain's own phase"
           if ok else "not needed - the row fails the loss gate")
    board = (
        f"\n## Amendment {AMEND[0]} (2026-11-23 {time.strftime('%H:%M')}): chain 19 point ({which}) "
        f"`{name}` - fixed-basis codec m={m}, {card:,}/{note:,} = {total:,} bits\n\n"
        f"Approach 1 (fixed-basis state coding) is the codec A/B. `{name}` holds row 17's card at "
        f"rank-2 int8 + int4 shifts and replaces the note's per-step rank-1 factors with an m x m "
        f"coefficient block against a basis that ships with the model: {card:,}/{note:,} = {total:,} "
        f"bits, {3584 - total} below row 17 and {abs(total - 2944)} {'above' if total > 2944 else 'below'} "
        f"row 45. Landed {when} after {eval_min:.1f} min of eval. It reads {a['mean_cur']:.6f}/"
        f"{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} (ub95 {fmt(a['ub95'])}/"
        f"{fmt(i['ub95'])}, 750 users) - {'PASS on both heads' if ok else '**FAIL**'}. The codec's "
        f"held-out energy capture is {capture} at m={m}, against the shipped rank-1 note's 0.904-0.951 "
        f"per layer; priced against row 17 the codec costs {fmt(vs17[0])}/{fmt(vs17[1])}. The row's "
        f"finiteness {fin}, and its speed cell stays UNMEASURED because the codec replaces "
        f"`compress_wkv_state` in the deploy engine.\n"
    )
    return row, para, note_bullet, board, ok


AMEND = [37]


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
    for row_id, name, stamp, capture, compare in POINTS:
        r = load(stamp)
        if r is None:
            print(f"{name}: no sweep yet ({stamp})")
            continue
        todo.append((row_id, name, stamp, build(row_id, name, stamp, capture, compare, r)))
        AMEND[0] += 1
    if not todo:
        print("nothing to record")
        return 0

    okall = True
    for row_id, name, stamp, (row, para, note_bullet, board, ok) in todo:
        print(f"verdict row {row_id} ({name}): {'PASS' if ok else 'FAIL'} "
              f"{' '.join(row.split())[:100]}...")
        if ok:
            print(f"  WARNING: row {row_id} passes the loss gate but its SPEED is unmeasured - "
                  f"it needs `rust_speed.py` before it is deployable")

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
            # the paragraph goes after the last paragraph of the live-narrative block
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

        def do_notes(lf: str, name=name, bullet=note_bullet):
            if f"`{name}` lands as RESULTS row" in lf:
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
    print("chain 19 points recorded" if okall else "SOMETHING FAILED")
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())
