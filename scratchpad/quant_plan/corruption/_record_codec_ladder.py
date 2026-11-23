"""Record the fixed-basis (codec) chains' points as RESULTS rows: chain 27 (`it36fbwidth`, the codec's
coefficient width + the card on the codec) and chain 28 (`it36shiftsb`, the token-shift codec ladder).

ONE SCRIPT, NINE POINTS. Both chains walk ONE knob at a time from a point that already PASSES, so each
row is a clean A/B: chain 27 holds row 17's card (points 1/2) or the codec's note at m=6 int8 (points
3/4) and moves the note's coefficient width, the card's re-presentation, or both; chain 28 holds chain
26's PASSING row 61 (`l2i8fb4n`, 1,280/1,152 = 2,432 bits) and changes ONLY the note's shift coder.

EVERY FIGURE IS READ AT RUN TIME from `quant_sweep_<stamp>.json` (bits, both heads' deltas/ub95, the
mean losses, the wall clock) and from the chain log's DONE line. Row ids are dynamic; a point whose row
already exists (recognised by its config name plus its own chain tag in the row) is skipped, so the
script is safe to re-run after each point lands. The speed cell comes from a `rust_speed_*.json` that
names the config (`c28` names chain 28's rungs; chain 27's four are in `c27`).

USAGE (after the sweep JSON exists; a missing sweep is skipped, never guessed):
  .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_codec_ladder.py [--only NAME]
"""
import argparse
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

CEIL = 0.0005
SPEED_FLOOR = 0.75

# the reference rows every comparison is quoted against: (label, bits, delta ahead, delta imm, what)
REFS = {
    "17": ("row 17", 3584, 0.000052, 0.000327, "the rank-2/rank-1 layout at 3,584 bits"),
    "45": ("row 45", 2944, 0.000125, 0.000381,
           "the uniform (rank, factor width, shift width) design's frontier"),
    "51": ("row 51", 2912, 0.000018, 0.000209, "the codec's own m=6 int8 rung"),
    "61": ("row 61", 2432, 0.000032, 0.000311,
           "chain 26 point 1 - the SAME layout without chain 28's shift coder"),
}

# (chain, chain tag, chain log, point index, config, stamp, card bits, note bits, layout, the knob,
#  the quality column, the reference rows to compare against)
POINTS = [
    (27, "it36fbwidth", "it36fbwidth_chain.log", 1, "l2i8fb6n6", "it36l2i8fb6n6", 1280, 1416,
     "card rank-2 int8 + note codec m=6 with int6 coefficients",
     "the note's coefficient block at int6 instead of int8 (m held at 6)",
     "held-out capture unchanged at m=6 (0.818-0.871)",
     ["45", "51", "61"]),
    (27, "it36fbwidth", "it36fbwidth_chain.log", 2, "l2i8fb6n4", "it36l2i8fb6n4", 1280, 1200,
     "card rank-2 int8 + note codec m=6 with int4 coefficients",
     "the note's coefficient block at int4 (16 levels per matrix entry)",
     "held-out capture unchanged at m=6 (0.818-0.871)",
     ["45", "51", "61"]),
    (27, "it36fbwidth", "it36fbwidth_chain.log", 3, "fbc8n6i8", "it36fbc8n6i8", 768, 1632,
     "card codec m=8 int8 + note codec m=6 int8",
     "the CARD leaves the rank-2 family for its own codec m=8 int8 (768 bits against 1,280)",
     "card held-out capture 0.9195 at m=8, note 0.818-0.871 at m=6",
     ["17", "45", "51"]),
    (27, "it36fbwidth", "it36fbwidth_chain.log", 4, "fbc8n6i6", "it36fbc8n6i6", 768, 1416,
     "card codec m=8 int8 + note codec m=6 int6",
     "the card codec AND the note's int6 coefficients together",
     "card held-out capture 0.9195 at m=8, note 0.818-0.871 at m=6",
     ["17", "45", "51", "61"]),
    (28, "it36shiftsb", "it36shiftsb_chain.log", 1, "l2i8fb4nsb16i4", "it36l2i8fb4nsb16i4", 1280, 768,
     "row 61's layout + the note's shift coder at m=16 int4",
     "the note's 256 shift bits per layer become 2*16*4 = 128, at rel err 0.01119 against raw int4's "
     "0.01214 at 128 bits/vec",
     "the note's shift reconstruction error is 0.01119 per vector against raw int4's 0.01214",
     ["61", "45"]),
    (28, "it36shiftsb", "it36shiftsb_chain.log", 2, "l2i8fb4nsb16i6", "it36l2i8fb4nsb16i6", 1280, 960,
     "row 61's layout + the note's shift coder at m=16 int6",
     "the shift coefficients at int6 (192 bits per layer) instead of int4 (128)",
     "the note's shift reconstruction error is 0.00664 per vector",
     ["61", "45"]),
    (28, "it36shiftsb", "it36shiftsb_chain.log", 3, "l2i8fb4nsb16i8", "it36l2i8fb4nsb16i8", 1280, 1152,
     "row 61's layout + the note's shift coder at m=16 int8 (the SAME-BITS control)",
     "the shift coefficients at int8, i.e. the SAME 256 bits per layer as row 61's raw int4, but "
     "1.9x lower error (0.00641 against 0.01214)",
     "the note's shift reconstruction error is 0.00641 per vector at unchanged bits",
     ["61", "45"]),
    (28, "it36shiftsb", "it36shiftsb_chain.log", 4, "l2i8fb4nsb12i4", "it36l2i8fb4nsb12i4", 1280, 672,
     "row 61's layout + the note's shift coder at m=12 int4",
     "m=12, i.e. 96 bits per layer (3/8 of the raw int4 budget) at rel err 0.01568",
     "the note's shift reconstruction error is 0.01568 per vector",
     ["61", "45"]),
    (28, "it36shiftsb", "it36shiftsb_chain.log", 5, "l2i8fb4nsb8i4", "it36l2i8fb4nsb8i4", 1280, 576,
     "row 61's layout + the note's shift coder at m=8 int4",
     "m=8, i.e. 64 bits per layer (1/4 of the raw int4 budget) at rel err 0.02486 - the LOW end of the "
     "ladder, where the population fit says the codec is worse than raw int4 at equal bits",
     "the note's shift reconstruction error is 0.02486 per vector",
     ["61", "45"]),
]

SPEED_TAGS = ["c28", "c27", "fbrung", "fb_speed3", "fb_idle"]


def speed_of(name):
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


def load(stamp):
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


def landing(logname, name):
    p = RUN / logname
    if not p.exists():
        return "landed", 0.0
    m = None
    for m in re.finditer(rf"\[(\d\d:\d\d:\d\d)\] DONE  eval {name}@124: rc=0 in ([\d.]+) min",
                         p.read_text(encoding="utf-8", errors="replace")):
        pass
    return (m.group(1)[:5], float(m.group(2))) if m else ("landed", 0.0)


def fmt(x):
    return f"{x:+.6f}"


def cmp_word(x):
    return f"BETTER by {abs(x):.6f}" if x < 0 else f"worse by {abs(x):.6f}"


def next_row(lf):
    ids = [int(n) for n in re.findall(r"^\| (\d+) \|", lf, re.M)]
    return (max(ids) + 1) if ids else 1


def next_amendment(lf):
    nums = [int(n) for n in re.findall(r"^## Amendment (\d+)", lf, re.M)]
    return max(nums) + 1 if nums else 1


def already_recorded(lf, name, tag):
    return any(re.match(r"^\| \d+ \|", l) and f"`{name}`" in l and tag in l
               for l in lf.split("\n"))


def build(row_id, chain, tag, logname, idx, name, stamp, card, note, layout, knob, quality, refs, r):
    total = card + note
    g = r["gate"]
    a, i = g["ahead"], g["imm"]
    pass_a, pass_i = a["ub95"] <= CEIL, i["ub95"] <= CEIL
    ok = bool(r["bits_ok"]) and pass_a and pass_i
    mins = r.get("secs", 0.0) / 60.0
    when, eval_min = landing(logname, name)
    speed = speed_of(name)
    if speed:
        rev, xrev, med, stag = speed
        sp_cells = f"{rev:,.1f} | {xrev:.3f}x"
        sp_ok = med >= SPEED_FLOOR
        sp_clause = (f"the deploy engine measures {rev:,.1f} rev/s = {xrev:.3f}x rev/s (median "
                     f"per-user {med:.3f}x) against `plain` in the same run (`rust_speed_{stag}.json`), "
                     f"{'clearing' if sp_ok else 'FAILING'} the 0.75x floor by "
                     f"{abs(med - SPEED_FLOOR):.3f}")
    else:
        med = None
        sp_ok = None
        sp_cells = "unmeasured | unmeasured"
        sp_clause = (f"UNMEASURED for `{name}` - `rust_speed.py --configs plain,{name}` is what would "
                     f"settle it")
    cmp_parts = []
    for key in refs:
        label, bits, ra, ri, _what = REFS[key]
        da, di = a["delta"] - ra, i["delta"] - ri
        verb = ("BETTER" if da < 0 else "worse") + "/" + ("BETTER" if di < 0 else "worse")
        cmp_parts.append(f"against {label} ({fmt(ra)}/{fmt(ri)}, {bits - total:+d} bits) {verb} "
                         f"{abs(da):.6f}/{abs(di):.6f}")
    cmp_line = "; ".join(cmp_parts)
    if cmp_line:
        cmp_line = cmp_line[0].upper() + cmp_line[1:]
    verdict = ("**PASS on both heads**" if ok else "**FAIL**") + \
        f": ahead ub95 {fmt(a['ub95'])} ({a['ub95'] / CEIL:.2f}x of the ceiling), imm ub95 " \
        f"{fmt(i['ub95'])} ({i['ub95'] / CEIL:.2f}x). {cmp_line}. {knob.capitalize()}, so the row " \
        f"{'is a deploy candidate' if ok else 'closes this rung of the ladder'}. Speed: {sp_clause}."
    row = (f"| {row_id} | QAT {layout} - chain {chain} point {idx}, `{name}` ({card:,}/{note:,} = "
           f"{total:,} bits; {quality}; landed {when}, {eval_min:.1f} min eval) | {card:,} ({card // 8}) "
           f"| {note:,} ({note // 8}) | {sp_cells} | {a['mean_cur']:.6f} | {i['mean_cur']:.6f} | "
           f"{fmt(a['delta'])} | {fmt(i['delta'])} | {verdict} | {'pending' if ok else '-'} |")
    para = (f"\n\n**Row {row_id} (`{name}`) is chain {chain}'s point {idx} ({layout}, {card:,}/{note:,} = "
            f"{total:,} bits).** {knob.capitalize()}. It reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
            f"{fmt(a['delta'])}/{fmt(i['delta'])} against `qnj_it36ctl` (ub95 {fmt(a['ub95'])}/"
            f"{fmt(i['ub95'])}, 750 users, {mins:.1f} min) - {'a PASS on both heads' if ok else 'a FAIL'}. "
            f"Quality column: {quality}. {cmp_line}. Speed: {sp_clause}.")
    bullet = (f"* {time.strftime('%Y-%m-%d %H:%M')} - **Chain {chain} point {idx} `{name}` lands as "
              f"RESULTS row {row_id}: {card:,}/{note:,} = {total:,} bits, {'PASS' if ok else 'FAIL'}.** "
              f"{knob.capitalize()}. It reads {fmt(a['delta'])}/{fmt(i['delta'])} vs `qnj_it36ctl` "
              f"(ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users, {mins:.1f} min). {cmp_line}. "
              f"Speed: {sp_clause}.\n")
    fin = ("a corruption check runs in this chain's own phase"
           if ok else "not needed - the row fails the loss gate")
    board = (f"\n## Amendment {AMEND[0]} (2026-11-23 {time.strftime('%H:%M')}): chain {chain} point "
             f"{idx} `{name}` - {layout}, {card:,}/{note:,} = {total:,} bits\n\n"
             f"{knob.capitalize()}. `{name}` reads {a['mean_cur']:.6f}/{i['mean_cur']:.6f} = "
             f"{fmt(a['delta'])}/{fmt(i['delta'])} (ub95 {fmt(a['ub95'])}/{fmt(i['ub95'])}, 750 users) - "
             f"{'PASS on both heads' if ok else '**FAIL**'}. {cmp_line}. Quality: {quality}. Speed: "
             f"{sp_clause}. Finiteness: {fin}.\n")
    return row, para, bullet, board, ok, sp_ok, med


AMEND = [0]


def edit(path, fn):
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
    print(f"{path.name}: written ({len(raw)} -> {len(out.encode('utf-8'))} bytes)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    AMEND[0] = next_amendment(BOARD.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n"))
    rl = RESULTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    row_id = next_row(rl)
    todo = []
    for spec in POINTS:
        chain, tag, logname, idx, name, stamp = spec[:6]
        if args.only and name != args.only:
            continue
        if already_recorded(rl, name, tag):
            print(f"{name}: already recorded - skipping")
            continue
        r = load(stamp)
        if r is None:
            print(f"{name}: no sweep yet ({stamp})")
            continue
        card, note = r["bits"]["card"], r["bits"]["note"]
        exp = (spec[6], spec[7])
        if (card, note) != exp:
            print(f"FAIL {name}: sweep says {card}/{note}, the registry says {exp} - refusing to guess")
            continue
        todo.append((row_id, spec, name, build(row_id, *spec[:6], card, note, *spec[8:], r)))
        row_id += 1
        AMEND[0] += 1
    if not todo:
        print("nothing to record")
        return 0
    okall = True
    for row_id, spec, name, (row, para, bullet, board, ok, sp_ok, med) in todo:
        print(f"row {row_id} ({name}): {'PASS' if ok else 'FAIL'}"
              f"{'' if sp_ok is None or not ok else (' speed %.3fx' % med)}")
        if ok and sp_ok is None:
            print(f"  WARNING: row {row_id} passes the loss gate but its speed is unmeasured")

        def do_results(lf, row_id=row_id, row=row, para=para):
            lines = lf.split("\n")
            data = [n for n, l in enumerate(lines) if re.match(r"^\| \d+ \|", l)]
            if not data:
                print("FAIL: no data rows")
                return None
            lines.insert(data[-1] + 1, row)
            out = "\n".join(lines)
            anchor = ("which holds the queue so the checks run in the first free window after the "
                      "chain's end marker.")
            if out.count(anchor) == 1:
                j = out.index(anchor) + len(anchor)
                out = out[:j] + para + out[j:]
            else:
                out = out.rstrip("\n") + para + "\n"
            got = [l for l in out.split("\n") if re.match(rf"^\|\s*{row_id}\s*\|", l)]
            if len(got) != 1 or len(got[0].split("|")) != 14:
                print(f"FAIL: row {row_id} malformed ({len(got)} hits, "
                      f"{len(got[0].split('|')) - 2 if got else 0} cells)")
                return None
            return out

        def do_notes(lf, bullet=bullet, name=name, tag=spec[1]):
            if f"`{name}` lands as RESULTS row" in lf:
                print("FAIL: NOTES bullet already present")
                return None
            m2 = re.search(r"^\* \d{4}-\d\d-\d\d \d\d:\d\d - ", lf, re.M)
            if not m2:
                print("FAIL: no NOTES anchor")
                return None
            return lf[:m2.start()] + bullet + lf[m2.start():]

        def do_board(lf, board=board, name=name):
            if re.search(rf": chain \d+ point \d+ `{re.escape(name)}`", lf):
                print(f"skip: the BOARD already carries a row for {name}")
                return lf
            n = next_amendment(lf)
            body = re.sub(r"^## Amendment \d+", f"## Amendment {n}", board.lstrip("\n"), count=1)
            return lf.rstrip("\n") + "\n" + body

        def do_recorded(lf, stamp=spec[5]):
            if f"{stamp}\n" in lf:
                print("FAIL: stamp already present")
                return None
            return lf.rstrip("\n") + f"\n{stamp}\n"

        okall = edit(RESULTS, do_results) and okall
        okall = edit(NOTES, do_notes) and okall
        okall = edit(BOARD, do_board) and okall
        okall = edit(RECORDED, do_recorded) and okall
    return 0 if okall else 1


if __name__ == "__main__":
    sys.exit(main())
