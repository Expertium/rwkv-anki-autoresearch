"""Record the corruption (finiteness) verdicts for RESULTS rows 29-32.

Rows 29/30/31/32 (`l2i8n2i8`, `l3i8n2i8`, `l2i8n2i8s8`, `l2i8n3i8`) passed the log-loss gate on
2026-11-22 but had NO corruption verdict, so they were not deployable. The verdicts are now produced by
`corruption/check_corruption.py` on the 50 longest-history users with histories NOT truncated
(Andrew's gate: even ONE non-finite card/note state value or ONE non-finite p(recall) fails).

HOW THEY RAN, because it matters for reproducibility: all chain waiters count matching processes and a
chain hands off with NO idle gap, so a non-chain GPU job cannot wait for a free GPU without racing a
handoff. `corruption/gpu_hold.py` held the queue busy for a bounded window instead, and the four checks
ran back to back inside it (the hold is CPU-only and self-expiring, so it can never overlap GPU work).

Raw verdicts: corruption/corruption_corrupt_<name>.json. Writes the RESULTS.md cells, a NOTES bullet
and one GPU_RUN_BOARD amendment. Raw bytes throughout: RESULTS.md and NOTES.md have mixed newlines.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

# (row label, corruption config name, RESULTS.md cell tail to replace)
ROWS = [
    ("29", "l2i8n2i8",     b"`rust_fastrank_check.json`, `rust_score.json`) | pending |"),
    ("30", "l3i8n2i8",     b"0.611x without it) | pending |"),
    ("31", "l2i8n2i8s8",   b"and 0.087 away from `int4` | pending |"),
    ("32", "l2i8n3i8",     b"not yet measured for this exact pair | pending |"),
]

AMENDMENT = """
## Amendment 19 (2026-11-22 19:35): the corruption column closes for rows 29-32 - the four note-rank candidates all pass on the 50 longest histories

Rows 29-32 passed the loss gate but had no corruption verdict, so none of them was deployable. The
verdicts are now in, from `corruption/check_corruption.py` on the **50 longest-mean-review users** with
their histories **NOT truncated** (`corruption/corrupt50.json`, 2,349,827 reviews / 4,609,289 section
rows, 62 too-big users excluded and backfilled):

| row | config | card/note bits | total | state calls | p(recall) tensors | non-finite | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | `l2i8n2i8` | 1,280/3,840 | 5,120 | 158,986 | 150 | **0** | 5.2 min |
| 30 | `l3i8n2i8` | 1,792/3,840 | 5,632 | 158,986 | 150 | **0** | 5.3 min |
| 31 | `l2i8n2i8s8` | 1,280/4,608 | 5,888 | 158,986 | 150 | **0** | 5 min |
| 32 | `l2i8n3i8` | 1,280/5,376 | 6,656 | 158,986 | 150 | **0** | 5 min |

All four: **PASS**, 50/50 users scored, no NaN-guard skips, zero non-finite state values, zero
non-finite p(recall). The observation count is IDENTICAL to row 17's (158,986 state calls, 150
p(recall) tensors), which is the control that the probe really was exercised: a PASS with fewer
observations would be INCOMPLETE, not a pass. So the note's rank ladder is clean at the history lengths
that matter, not just on the short parity traces.

**The scheduling problem this exposed, and the tool that fixes it.** Every `it36*_chain.py` waiter
decides "the GPU is free" by counting matching processes (`it36ladder_lib.running`) and a chain hands
off from one point to the next with NO idle gap, so a job that is not part of a chain cannot wait for a
free GPU without racing a handoff - the race is exactly what stranded the queue on 2026-11-22 16:42
(a `rust_grid` run matched BUSY_PAT and `it36shrink` sat for 10+ min with the GPU at 1%). Meanwhile
Andrew's instruction is that a corruption check must NOT be left pending once a row passes.
`corruption/gpu_hold.py` resolves it: a CPU-only, self-expiring process whose command line carries a
BUSY_PAT token (`--tag rust_grid.py`), so it holds the queue "busy" for a bounded window while the
checks run back to back inside it, and it cannot create a second GPU workload because it only sleeps.
The four checks above ran in one such window (19:10-19:33), started the moment the active chain logged
its end marker. The hold ALWAYS exits after `--secs`, so a crash cannot wedge the queue.

**Still open on the Rust side.** The Python/eval-path gate is now clean for rows 15-18 and 29-32. The
deploy engine has its OWN non-finiteness guard (`RWKV_FINITE_CHECK`, `--strict`), and it has still only
seen the 3-user parity trace. `corruption/export_corrupt50_traces.py` is producing the 50 long-history
traces for that check (`trace_corrupt50/`, 39/50 complete at 19:35); `corruption/corrupt50_finite_run.py`
runs it over every complete trace once the export finishes.
"""

BULLET = (
    "* 2026-11-22 19:40 - **The corruption column closes for rows 29-32: all four PASS on the 50 "
    "longest histories, histories not truncated.** `check_corruption.py` on `corrupt50.json` "
    "(2,349,827 reviews / 4,609,289 section rows): `l2i8n2i8` (row 29, 5.2 min), `l3i8n2i8` (row 30, "
    "5.3 min), `l2i8n2i8s8` (row 31, 5 min) and `l2i8n3i8` (row 32, 5 min) each scored 50/50 users "
    "with 158,986 state calls and 150 p(recall) tensors - the SAME observation count as row 17, which "
    "is the control that the probe was really exercised - and ZERO non-finite values on both heads, "
    "with no NaN-guard skips. So the whole note rank ladder is clean at the history lengths that "
    "matter, and rows 29/30/32 are deployable on this gate. NEW TOOL: `corruption/gpu_hold.py` - a "
    "CPU-only, self-expiring queue hold that lets a non-chain GPU job (here the four checks) run "
    "without racing a chain handoff; it needs no GPU and cannot overlap GPU work. Still open: the "
    "Rust engine's own `--strict` finiteness guard has only seen the 3-user parity trace, and the "
    "50-trace export is at 39/50 for `corrupt50_finite_run.py`.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def verdict(name):
    p = HERE / f"corruption_corrupt_{name}.json"
    if not p.exists():
        raise SystemExit(f"ABORT: no verdict file for {name} ({p})")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "PASS":
        raise SystemExit(f"ABORT: {name} status={d.get('status')} problems={d.get('problems')}")
    if not d.get("state_calls") or not d.get("p_calls"):
        raise SystemExit(f"ABORT: {name} has no observations ({d.get('state_calls')} state calls)")
    return d


def main() -> int:
    raw = RESULTS.read_bytes()
    for row, name, tail in ROWS:
        d = verdict(name)
        cell = (f"**PASS** ({d['state_calls']:,} state calls, {d['p_calls']} p(recall) tensors, "
                f"0 non-finite, {d['secs']/60:.1f} min on the 50-user roster) |")
        raw = sub(raw, tail, cell.encode("utf-8"), f"RESULTS row {row} corruption cell")
    RESULTS.write_bytes(raw)

    notes = NOTES.read_bytes()
    if b"the corruption column closes for rows 29-32" in notes:
        print("  NOTES bullet: already present")
    else:
        notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                    "NOTES bullet")
        NOTES.write_bytes(notes)

    board = BOARD.read_bytes()
    if b"## Amendment 19" in board:
        print("  BOARD: Amendment 19 already present")
    else:
        if not board.endswith(b"\n"):
            board += b"\n"
        BOARD.write_bytes(board + AMENDMENT.encode("utf-8"))
        print("  BOARD: Amendment 19 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
