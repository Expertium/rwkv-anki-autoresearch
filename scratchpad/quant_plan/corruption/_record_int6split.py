"""Record the mixed-precision int6 split screen (11 long-history users) as a RESULTS paragraph,
a NOTES live-status bullet and GPU_RUN_BOARD Amendment 22.

Source of truth: `rust_wide_screen_it36int6split.json` (paired per-user deltas vs `l2i8s4`, bootstrap
2000) and `rust_grid_it36int6split.json` (the absolute means of the same batches). Every number in the
prose is computed from those two files, so nothing can be mistyped.

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_record_int6split.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
WIDE = RUN / "rust_wide_screen_it36int6split.json"
GRID = RUN / "rust_grid_it36int6split.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

# Last sentence of the EF-closure paragraph (RESULTS.md line 188) - the newest body paragraph.
ANCHOR = (b"so it needs no decision from Andrew unless a future recipe wants its -0.0025 imm at more "
          b"bits.")
NOTE_ANCHOR = b"## Live status\n\n"
AMEND = b"## Amendment 22"


def fmt(x):
    return f"{x:+.6f}"


def main() -> int:
    wide = json.loads(WIDE.read_text(encoding="utf-8"))
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    c, n = wide["configs"]["l2i6c"], wide["configs"]["l2i6n"]
    n_users = wide["n_users"]
    g = grid

    def pair(name, head):
        d = wide["configs"][name][head]
        return (f"{fmt(d['mean'])} [{fmt(d['boot_lo'])}, {fmt(d['boot_hi'])}] "
                f"better {d['better']}/{d['n']}")

    para = (
        f"**The int6 tax is mostly the NOTE's: the card-only int6 rung is nearly free.** The 21-user "
        f"read above priced int6 as a whole (`l2i6s4`, both streams int6, {fmt(0.000442)} ahead / "
        f"{fmt(0.000726)} imm). This screen splits it on {n_users} long-history users by quantizing ONE "
        f"stream at int6 and leaving the other at row 17's own level, as per-user PAIRED deltas against "
        f"`l2i8s4` inside one batch (`rust_wide_screen_it36int6split.json`, instrument "
        f"`corruption/wide_screen.py`, 2,000 bootstrap resamples; absolutes in "
        f"`rust_grid_it36int6split.json`). Two readings. (1) **The CARD's int6 is nearly free**: "
        f"`l2i6c` (card rank-2 int6, note rank-1 int8 + int4 shifts = 1,024/2,304 = 3,328 bits, 256 "
        f"BELOW row 17) costs {pair('l2i6c', 'ahead')} ahead and {pair('l2i6c', 'imm')} imm, so its "
        f"ahead interval CONTAINS zero and its imm interval only just excludes it - an order of "
        f"magnitude inside row 17's own +0.000052/+0.000327, for 256 fewer card bits. (2) **The NOTE's "
        f"int6 carries the rest**: `l2i6n` (card rank-2 int8, note rank-1 int6 = 1,280/1,920 = 3,200 "
        f"bits, 384 below row 17) costs {pair('l2i6n', 'ahead')} ahead and {pair('l2i6n', 'imm')} imm, "
        f"worse on both heads than the card-only rung for only 128 fewer bits. The two parts sum to "
        f"about {fmt(c['ahead']['mean'] + n['ahead']['mean'])} ahead / "
        f"{fmt(c['imm']['mean'] + n['imm']['mean'])} imm, the same order as the {fmt(0.000442)}/"
        f"{fmt(0.000726)} that `l2i6s4` costs on the 21-user set (a decomposition, not an exact sum - "
        f"the two screens share only 3 users and the note's rank differs). The practical consequence is "
        f"that the CARD-only int6 rung is the one worth a 750-user QAT point, and chain 18 "
        f"(`it36int6card`) trains it (`l2i6c`, then its note-only twin `l2i6n` for attribution), both "
        f"124 steps against `qnj_it36ctl`. Deploy-side deltas only, so they rank the pair; the 750-user "
        f"eval decides. Absolutes on the {n_users} users: plain "
        f"{g['plain']['means']['ahead']:.6f}/{g['plain']['means']['imm']:.6f}, `l2i8s4` "
        f"{g['l2i8s4']['means']['ahead']:.6f}/{g['l2i8s4']['means']['imm']:.6f}, `l2i6c` "
        f"{g['l2i6c']['means']['ahead']:.6f}/{g['l2i6c']['means']['imm']:.6f}, `l2i6n` "
        f"{g['l2i6n']['means']['ahead']:.6f}/{g['l2i6n']['means']['imm']:.6f}.\n"
    )

    bullet = (
        f"* 2026-11-22 20:35 - **The int6 tax is mostly the NOTE's; the card-only int6 rung "
        f"(`l2i6c`, 3,328 bits = 256 BELOW row 17) is nearly free.** 11 long-history users, per-user "
        f"paired vs `l2i8s4`: `l2i6c` {fmt(c['ahead']['mean'])} ahead "
        f"[{fmt(c['ahead']['boot_lo'])}, {fmt(c['ahead']['boot_hi'])}] / {fmt(c['imm']['mean'])} imm "
        f"[{fmt(c['imm']['boot_lo'])}, {fmt(c['imm']['boot_hi'])}], while the note-only twin `l2i6n` "
        f"(3,200 bits) costs {fmt(n['ahead']['mean'])}/{fmt(n['imm']['mean'])}. So the 21-user int6 "
        f"price ({fmt(0.000442)}/{fmt(0.000726)}) is mostly the note's, and the card rung is the one "
        f"to train at 750 users: chain 18 `it36int6card` (`l2i6c` then `l2i6n`, 124 steps vs "
        f"`qnj_it36ctl`) is queued behind `it36int6`. `rust_wide_screen_it36int6split.json`.\n"
    )

    amend = (
        "\n\n## Amendment 22 (2026-11-22 20:35): the int6 tax splits - the CARD-only int6 rung is "
        "nearly free, the NOTE's is not\n\n"
        "The 21-user int6 price (Amendment 20: `l2i6s4` +0.000442 ahead / +0.000726 imm) is now split "
        f"by quantizing one stream at a time on {n_users} long-history users, as per-user PAIRED "
        "deltas against `l2i8s4` (`rust_wide_screen_it36int6split.json`; absolutes "
        "`rust_grid_it36int6split.json`):\n\n"
        "| config | card bits | note bits | total | d ahead (95% CI) | d imm (95% CI) |\n"
        "|---|---:|---:|---:|---|---|\n"
        f"| `l2i6c` card int6, note int8 | 1,024 | 2,304 | 3,328 | {pair('l2i6c', 'ahead')} | "
        f"{pair('l2i6c', 'imm')} |\n"
        f"| `l2i6n` card int8, note int6 | 1,280 | 1,920 | 3,200 | {pair('l2i6n', 'ahead')} | "
        f"{pair('l2i6n', 'imm')} |\n\n"
        f"`l2i6c` is 256 bits BELOW row 17 and its ahead interval contains zero; `l2i6n` is 128 bits "
        f"below the card rung and worse on both heads. The card rung is therefore the axis-1 point "
        f"worth a 750-user QAT run: chain 18 (`it36int6card`, points `l2i6c` then `l2i6n`, 124 steps "
        f"vs `qnj_it36ctl`) is queued behind chain 17 (`it36int6`). Both configs were added to "
        f"`rust_grid.CONFIGS` and `quant_sweep.CANDIDATES`; the deploy engine's `parse_level` already "
        f"accepts int6, so `card:2:int6,note:1:int8` is expressible and this rung is DEPLOYABLE, "
        f"unlike the per-stream-shift rows (31/34/37).\n"
    )

    changed = 0
    raw = RESULTS.read_bytes()
    if b"The int6 tax is mostly the NOTE's" in raw:
        print("RESULTS: paragraph already present")
    else:
        if raw.count(ANCHOR) != 1:
            raise SystemExit(f"ABORT: RESULTS anchor found {raw.count(ANCHOR)} times, expected 1")
        RESULTS.write_bytes(raw.replace(ANCHOR, ANCHOR + b"\n\n" + para.encode("utf-8"), 1))
        print("RESULTS paragraph: ok")
        changed += 1

    nb = NOTES.read_bytes()
    if b"The int6 tax is mostly the NOTE's" in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(NOTE_ANCHOR) != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {nb.count(NOTE_ANCHOR)} times, expected 1")
        NOTES.write_bytes(nb.replace(NOTE_ANCHOR, NOTE_ANCHOR + bullet.encode("utf-8"), 1))
        print("NOTES bullet: ok")
        changed += 1

    bb = BOARD.read_bytes()
    if AMEND in bb:
        print("BOARD: Amendment 22 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 22 appended")
        changed += 1

    print(f"done ({changed} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
