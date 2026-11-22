"""Record the deploy engine's finiteness coverage run (RESULTS paragraph + NOTES bullet + Amendment 21).

Data: `rust_finite_corrupt50.json` / `rust_finite_corrupt50_verdict.json`, produced by
`corruption/corrupt50_finite_run.py --tag corrupt50` (40 complete `trace_corrupt50` traces, the
driver's minimum) with `--strict`, so rc=0 is the ENGINE's own statement.
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
SRC = RUN / "rust_finite_corrupt50.json"

ANCHOR = b"so it was not re-run."


def totals():
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    c = doc["configs"]
    n = len(c["l2i8s4"]["per_user"])
    s = lambda k, f: sum(v[f] for v in c[k]["per_user"].values())
    return {
        "n": n,
        "plain_p": s("plain", "p_vals"), "plain_pbad": s("plain", "p_bad"),
        "comp": s("l2i8s4", "compressions"), "state": s("l2i8s4", "state_vals"),
        "vec": s("l2i8s4", "vec_vals"), "p": s("l2i8s4", "p_vals"),
        "bad": (s("l2i8s4", "state_raw_bad") + s("l2i8s4", "state_quant_bad")
                + s("l2i8s4", "vec_raw_bad") + s("l2i8s4", "vec_quant_bad")
                + s("l2i8s4", "p_bad")),
        "max_state": max(v["state_maxabs"] for v in c["l2i8s4"]["per_user"].values()),
        "max_vec": max(v["vec_maxabs"] for v in c["l2i8s4"]["per_user"].values()),
    }


def main() -> int:
    t = totals()
    if t["bad"]:
        raise SystemExit(f"ABORT: {t['bad']} non-finite values found -- do not record this as a pass")

    para = (
        "\n\n**The deploy engine's own finiteness guard clears 40 long histories - 7.2 billion state "
        "values, zero non-finite.** Andrew's blowup gate was already clean in PYTHON (rows 15-18 and "
        "29-32). The deploy engine has its OWN guard (`RWKV_FINITE_CHECK`, `--strict`, exit 3 on any "
        "non-finite value) and it had only ever seen the 3-user parity trace, so the shipped guard had "
        "never been exercised at the history lengths the gate is about. The `trace_corrupt50` export "
        "reached the driver's minimum of 40 complete traces, and "
        "`corruption/corrupt50_finite_run.py --tag corrupt50` ran over all of them: **rc=0**, i.e. the "
        f"engine itself found nothing. The `l2i8s4` (row 17) config observed {t['comp']:,} state "
        f"compressions and {t['state']:,} state values, {t['vec']:,} shift-vector values and "
        f"{t['p']:,} p(recall) values, on the RAW and the QUANTIZED side of every compression, with "
        f"ZERO bad values of any kind (largest |state| {t['max_state']:.3f}, largest |shift vector| "
        f"{t['max_vec']:.3f}); `plain` observed {t['plain_p']:,} p(recall) values, zero bad, and no "
        "state values at all because nothing is quantized there. On histories of up to 95,597 reviews "
        "this is roughly 45,000x the Python gate's 158,986 state calls on the same roster, so the two "
        "implementations now agree at the lengths that matter. The last 10 traces were still exporting; "
        "re-run with `--tag corrupt50_rest` when they land.\n"
    )

    bullet = (
        "* 2026-11-22 20:10 - **The deploy engine's finiteness guard now covers 40 long histories: "
        f"{t['state']:,} state values, ZERO non-finite.** `corruption/corrupt50_finite_run.py --tag "
        "corrupt50` (40 complete `trace_corrupt50` traces, `--strict`, so rc=0 is the engine's own "
        f"statement): `l2i8s4` observed {t['comp']:,} compressions, {t['state']:,} state values, "
        f"{t['vec']:,} shift-vector values and {t['p']:,} p(recall) values with zero bad on both the "
        f"raw and the quantized side; `plain` {t['plain_p']:,} p(recall) values, zero bad. That closes "
        "the coverage gap that existed because the guard had only seen the 3-user parity trace - the "
        "Python gate (158,986 state calls on the same 50-user roster) and the shipped engine now agree. "
        "10 traces were still exporting; re-run with `--tag corrupt50_rest`.\n"
    )

    raw = RESULTS.read_bytes()
    if b"finiteness guard clears 40 long histories" in raw:
        print("  RESULTS paragraph: already present")
    else:
        n = raw.count(ANCHOR)
        if n != 1:
            raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
        RESULTS.write_bytes(raw.replace(ANCHOR, ANCHOR + para.encode("utf-8"), 1))
        print("  RESULTS paragraph: ok")

    notes = NOTES.read_bytes()
    if b"finiteness guard now covers 40 long histories" in notes:
        print("  NOTES bullet: already present")
    else:
        n = notes.count(b"## Live status\n\n")
        if n != 1:
            raise SystemExit(f"ABORT: NOTES anchor found {n} times, expected 1")
        NOTES.write_bytes(notes.replace(b"## Live status\n\n",
                                        b"## Live status\n\n" + bullet.encode("utf-8"), 1))
        print("  NOTES bullet: ok")

    board = BOARD.read_bytes()
    if b"## Amendment 21" in board:
        print("  BOARD: Amendment 21 already present")
        return 0
    if not board.endswith(b"\n"):
        board += b"\n"
    amendment = f"""
## Amendment 21 (2026-11-22 20:10): the deploy engine's own finiteness guard clears 40 long histories

Andrew's blowup gate was clean in PYTHON first (rows 15-18 and 29-32, `check_corruption.py` on the
50 longest-mean-history users). The DEPLOY engine has its own guard (`RWKV_FINITE_CHECK` with
`--strict`, which exits 3 on any non-finite value) and it had only ever seen the 3-user parity trace,
so the implementation that actually ships had never been exercised at the history lengths the gate is
about. `corruption/export_corrupt50_traces.py` reached the driver's minimum of 40 complete traces, and
`corruption/corrupt50_finite_run.py --tag corrupt50` ran over all of them inside the active chain's
point window (CPU-only engine, so no GPU contention).

| config | compressions | state values | shift-vector values | p(recall) values | non-finite |
| --- | --- | --- | --- | --- | --- |
| `plain` | 0 (nothing quantized) | 0 | 0 | {t['plain_p']:,} | **0** |
| `l2i8s4` (row 17) | {t['comp']:,} | {t['state']:,} | {t['vec']:,} | {t['p']:,} | **0** |

`rc=0` for both configs over all 40 users: zero bad values in every per-user record, on the RAW and
the QUANTIZED side of every state compression and on every shift vector, plus zero planted failures
(the guard's own self-test). Largest |state| {t['max_state']:.3f}, largest |shift vector|
{t['max_vec']:.3f} - i.e. the states are small and stay small; nothing is near an overflow.

On histories of up to 95,597 reviews this is about 45,000x the Python gate's 158,986 state calls on the
same roster, so the two implementations agree where it matters. Note: the engine's `[finite] TOTAL`
line reports zeros because its cumulative counter does not aggregate across users - the per-user
records are the numbers above. The last 10 traces were still exporting; re-run with
`--tag corrupt50_rest` when they land.
"""
    BOARD.write_bytes(board + amendment.encode("utf-8"))
    print("  BOARD: Amendment 21 appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
