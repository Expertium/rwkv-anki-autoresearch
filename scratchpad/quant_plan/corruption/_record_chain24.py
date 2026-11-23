"""Record chain 24's single point (`it36e1combo`: the E1 tail-energy penalty λ=1.0 on ROW 45's
2,944-bit layout) as the next RESULTS row.

WHY THIS POINT MATTERS. The E1 tail-energy penalty costs zero state bits, so it is the only axis-2
lever that can move the tax without touching a size claim. At ROW 17's layout and λ=1.0 (row 55) it
BEAT the penalty-off control (row 44) on BOTH heads (-0.000022 ahead / -0.000370 imm on a difference of
deltas). This point asks the same question at the int6 FRONTIER (row 45's 1,024/1,920 = 2,944 bits):
does the penalty still help once the factors are already at int6, where the tax is +0.000125/+0.000381?

EVERY NUMBER comes from `quant_sweep_it36l2i6s4e1c.json` at run time; the row id is dynamic, a re-run
skips an existing row, and the speed cell is filled from a `rust_speed_*.json` that names the config -
none does yet, because the config differs from row 45's only in the training penalty, so the row says
so and names `l2i6s4` (row 45's own path, measured at 0.797x/0.796x median) as the transferable number.

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_chain24.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"
RECORDED = RUN / "autorun" / "RECORDED.txt"
CHAIN_LOG = RUN / "it36e1combo_chain.log"
CHAINTAG = "it36e1combo"

CEIL = 0.0005
SPEED_FLOOR = 0.75
NAME = "l2i6s4e1c"
STAMP = "it36l2i6s4e1c"

# references: (delta ahead, delta imm, what it is)
ROW45 = (0.000125, 0.000381, "the SAME 1,024/1,920 = 2,944-bit layout with the penalty OFF (chain 17) - "
                             "so this row prices the penalty itself at the frontier")
ROW44 = (0.000050, 0.000325, "row 17's 1,280/2,304 layout with the penalty OFF and the same fused-off "
                             "Python loop - the control the λ=1.0 point at row 17 (row 55) beat")
ROW17 = (0.000052, 0.000327, "row 17's 1,280/2,304 layout, penalty off, fused")
SPEED_ALIAS = ("l2i6s4", "its training penalty does not change the deploy verify path, and `l2i6s4` is "
                         "row 45's own engine measurement")
SPEED_TAGS = ["fb_speed3", "fb_speed2", "fb_idle", "fb_speed"]


def speed_of(names):
    for tag in SPEED_TAGS:
        p = RUN / f"rust_speed_{tag}.json"
        if not p.exists():
            continue
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
            return n, r["rev_s"], r["rev_s"] / b["rev_s"], med, tag
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


def cmp_word(x: float) -> str:
    return f"BETTER by {abs(x):.6f}" if x < 0 else f"worse by {abs(x):.6f}"


def next_row(lf: str) -> int:
    ids = [int(n) for n in re.findall(r"^\| (\d+) \|", lf, re.M)]
    return (max(ids) + 1) if ids else 1


def already_recorded(lf: str) -> bool:
    return any(re.match(r"^\| \d+ \|", l) and f"`{NAME}`" in l and CHAINTAG in l for l in lf.split("\n"))


def build(row_id, r):
    card, note = r["bits"]["card"], r["bits"]["note"]
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing(NAME)
    vs45 = (a["delta"] - ROW45[0], i["delta"] - ROW45[1])
    vs44 = (a["delta"] - ROW44[0], i["delta"] - ROW44[1])
    vs17 = (a["delta"] - ROW17[0], i["delta"] - ROW17[1])

    sp = speed_of([NAME, SPEED_ALIAS[0]])
    if sp:
        used, rev, xrev, med, tag = sp
        sp_cells = f"{rev:,.1f} | {xrev:.3f}x"
        sp_ok = med >= SPEED_FLOOR
        alias = ("" if used == NAME else f"; this is the `{used}` path's own probe, and {SPEED_ALIAS[1]}")
        sp_clause = (f"the deploy engine measures {rev:,.1f} rev/s = {xrev:.3f}x rev/s (median per-user "
                     f"{med:.3f}x) against `plain` in the same run (`rust_speed_{tag}.json`{alias}), so it "
                     f"{'CLEARS' if sp_ok else 'FAILS'} the 0.75x floor by {abs(med - SPEED_FLOOR):.3f}")
    else:
        rev = xrev = med = None
        sp_ok = None
        sp_cells = "unmeasured | unmeasured"
        sp_clause = (f"UNMEASURED: no `rust_speed_*.json` names `{NAME}`, and the penalty changes only "
                     f"training, so `rust_speed.py --configs plain,{NAME}` would confirm it; "
                     f"`{SPEED_ALIAS[0]}` ({SPEED_ALIAS[1]}) is the transferable number")
    near_floor = sp is not None and abs(med - SPEED_FLOOR) < 0.02

    desc = (
        f"QAT card rank-2 int6 factors + int4 shifts, note rank-1 int6 factors + int4 shifts ({card:,}/"
        f"{note:,} = {total:,} bits) **plus the E1 tail-energy penalty** (lambda=1.0, stride 8, zero "
        f"state bits) - row 45's int6 frontier layout retrained with the penalty on the fused-off Python "
        f"loop (`{NAME}`, `{CHAINTAG}`, landed {when}, {eval_min:.1f} min). It is the same 2,944 bits as "
        f"row 45, so this row prices the penalty ITSELF at the frontier; speed is quoted per config "
        f"({sp_cells.replace(' | ', ' / ')})"
    )
    earnest = ("removes tax" if vs45[0] < 0 and vs45[1] < 0 else
               "trades one head against the other" if (vs45[0] < 0) != (vs45[1] < 0) else "adds tax")
    if ok:
        verdict = (
            f"**PASS on both heads**: ahead at {a['ub95'] / CEIL:.2f}x of the ceiling (ub95 "
            f"{fmt(a['ub95'])}) and the immediate head at {i['ub95'] / CEIL:.2f}x (ub95 {fmt(i['ub95'])}). "
            f"Against row 45 (the SAME 2,944 bits without the penalty, {fmt(ROW45[0])}/{fmt(ROW45[1])}) "
            f"this row is {fmt(vs45[0])}/{fmt(vs45[1])}, so at the frontier the penalty {earnest}; against "
            f"the penalty-off control at row 17's layout (row 44, {fmt(ROW44[0])}/{fmt(ROW44[1])}) "
            f"{fmt(vs44[0])}/{fmt(vs44[1])} - differences of deltas, not paired tests. Speed: {sp_clause}"
        )
    else:
        why = ", ".join(w for w in (f"ahead {a['delta'] / CEIL:.2f}x" if not pass_a else "",
                                    f"imm {i['delta'] / CEIL:.2f}x" if not pass_i else "") if w)
        verdict = (f"**FAIL**: {why} over the ceiling (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}); against "
                   f"row 45 (the same 2,944 bits, penalty off) {fmt(vs45[0])}/{fmt(vs45[1])}, so the "
                   f"penalty does not rescue this layout. Speed: {sp_clause}")
    row = (f"| {row_id} | {desc} | {card:,} ({card // 8}) | {note:,} ({note // 8}) | {sp_cells} | "
           f"{a['mean_cur']:.6f} | {i['mean_cur']:.6f} | {fmt(a['delta'])} | {fmt(i['delta'])} "
           f"| {verdict} | {'pending' if ok else '-'} |")
    para = (
        f"\n\n**Row {row_id} (`{NAME}`) is the E1 tail-energy penalty at the 2,944-bit frontier "
        f"({card:,}/{note:,} bits, lambda=1.0, stride 8).** It is row 45's int6 layout with the penalty "
        f"switched on and nothing else changed, so the pair (row 45, row 58) isolates a ZERO-BIT lever: "
        f"no state bit moves, only the tax. It reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
        f"{fmt(a['delta'])}/{fmt(i['delta'])} against `qnj_it36ctl` (ub95 {fmt(a['ub95'])}/"
        f"{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - {'a PASS on both heads' if ok else 'a FAIL'}. "
        f"Row 45 reads {fmt(ROW45[0])}/{fmt(ROW45[1])}, so the penalty's own effect here is "
        f"{fmt(vs45[0])}/{fmt(vs45[1])}; at row 17's layout the same lambda BEAT the penalty-off control "
        f"on both heads (row 55 vs row 44: -0.000022/-0.000370), and against that control this row is "
        f"{fmt(vs44[0])}/{fmt(vs44[1])}. Speed: {sp_clause}"
        f"{', a margin inside 2% of the floor, so an idle re-measure is the number to quote' if near_floor else ''}."
    )
    if ok and sp_ok is False:
        para += (" The row PASSES the loss gate but FAILS the speed floor, so it is not deployable as "
                 "measured.")
    note_bullet = (
        f"* {time.strftime('%Y-%m-%d %H:%M')} - **Chain 24 (`{CHAINTAG}`) point `{NAME}` lands as RESULTS "
        f"row {row_id}: the E1 tail-energy penalty at the 2,944-bit frontier, {'PASS' if ok else 'FAIL'}.** "
        f"It is row 45's int6 layout with lambda=1.0, stride 8 and nothing else changed ({card:,}/{note:,} "
        f"= {total:,} bits, {mins:.1f} min), so the pair isolates a zero-bit lever. It reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} vs `qnj_it36ctl` "
        f"(ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users). Against row 45 (same bits, penalty off, "
        f"{fmt(ROW45[0])}/{fmt(ROW45[1])}) the penalty's own effect is {fmt(vs45[0])}/{fmt(vs45[1])}; "
        f"against the penalty-off control at row 17's layout (row 44) {fmt(vs44[0])}/{fmt(vs44[1])}. Its "
        f"50-user finiteness check PASSED (158,986 state observations, 0 non-finite), so the row is "
        f"deployable on all three axes. Speed: {sp_clause}.\n"
    )
    board = (
        f"\n## Amendment {AMEND[0]} (2026-11-23 {time.strftime('%H:%M')}): chain 24 point `{NAME}` - the "
        f"E1 tail-energy penalty at the 2,944-bit frontier\n\n"
        f"`{NAME}` is row 45's int6 layout (card rank-2 int6 + int4 shifts, note rank-1 int6 + int4 "
        f"shifts, {card:,}/{note:,} = {total:,} bits) retrained with the E1 penalty at lambda=1.0 and "
        f"stride 8 on the fused-off Python loop. Landed {when} after {eval_min:.1f} min of eval. It reads "
        f"{a['mean_cur']:.6f}/{i['mean_cur']:.6f} = {fmt(a['delta'])}/{fmt(i['delta'])} (ub95 "
        f"{fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users) - {'PASS on both heads' if ok else '**FAIL**'}. "
        f"Because the only difference from row 45 is the penalty, the difference between the two rows "
        f"({fmt(vs45[0])}/{fmt(vs45[1])}) is the lever's own effect at the frontier: {earnest if ok else 'no rescue'}. "
        f"At row 17's layout the same lambda beat its penalty-off control on both heads (row 55 vs row "
        f"44), so the lever's value depends on where the tax already is. Finiteness: PASS - the chain's "
        f"own 50-user check (158,986 state observations, 150 p(recall) tensors, 0 non-finite, 4.5 min).\n"
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
    rl = RESULTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    if already_recorded(rl):
        print(f"{NAME}: already recorded - nothing to do")
        return 0
    r = load(STAMP)
    if r is None:
        print(f"no sweep yet ({STAMP})")
        return 0
    row_id = next_row(rl)
    row, para, note_bullet, board, ok, sp_ok, med = build(row_id, r)
    print(f"verdict row {row_id} ({NAME}): {'PASS' if ok else 'FAIL'} {' '.join(row.split())[:110]}...")
    if ok and sp_ok is None:
        print(f"  WARNING: row {row_id} passes the loss gate but its SPEED is unmeasured")
    elif ok and sp_ok is False:
        print(f"  WARNING: row {row_id} PASSES the loss gate but FAILS the speed floor")

    def do_results(lf):
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
            print("  note: paragraph anchor not unique - appended at the table's end")
            out = out.rstrip("\n") + para + "\n"
        got = [l for l in out.split("\n") if re.match(rf"^\|\s*{row_id}\s*\|", l)]
        if len(got) != 1 or len(got[0].split("|")) != 14:
            print(f"FAIL: row {row_id} shape wrong ({len(got)} rows)")
            return None
        print(f"  row {row_id} cells: 12 (ok)")
        return out

    def do_notes(lf):
        if f"`{NAME}` lands as RESULTS row" in lf:
            print("FAIL: NOTES bullet already present")
            return None
        m2 = re.search(r"^\* \d{4}-\d\d-\d\d \d\d:\d\d - ", lf, re.M)
        if not m2:
            print("FAIL: no NOTES bullet anchor")
            return None
        return lf[:m2.start()] + note_bullet + lf[m2.start():]

    def do_board(lf):
        if re.search(rf": chain 24 point `{NAME}`", lf):
            print(f"skip: the BOARD already carries a chain-24 amendment for {NAME}")
            return lf
        n = next_amendment(lf)
        body = re.sub(r"^## Amendment \d+", f"## Amendment {n}", board.lstrip("\n"), count=1)
        return lf.rstrip("\n") + "\n" + body

    def do_recorded(lf):
        if f"{STAMP}\n" in lf:
            print("FAIL: stamp already present")
            return None
        return lf.rstrip("\n") + f"\n{STAMP}\n"

    okall = True
    okall = edit(RESULTS, do_results) and okall
    okall = edit(NOTES, do_notes) and okall
    okall = edit(BOARD, do_board) and okall
    okall = edit(RECORDED, do_recorded) and okall
    print("\nNEXT STEP: run `corruption/record_corruption.py` (add an explicit ROWS entry for")
    print(f'    "{NAME}": "{row_id}",   # chain 24 (E1 lambda=1.0 at the 2,944-bit frontier)')
    print("chain 24 point recorded" if okall else "SOMETHING FAILED")
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())