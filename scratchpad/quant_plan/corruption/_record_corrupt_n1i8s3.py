"""Record the `n1i8s3` (row 35) corruption verdict: PASS, and the corruption column is now CLOSED.

Source of truth: scratchpad/quant_plan/corruption/corruption_corrupt_n1i8s3.json (written 23:26:01).
The RESULTS cell itself was written by `record_corruption.py` (the rule is: never hand-edit a
corruption cell); this script updates the prose around it, the NOTES bullet and GPU_RUN_BOARD.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
VERD = RUN / "corruption" / "corruption_corrupt_n1i8s3.json"
RESULTS = RUN / "RESULTS.md"
NOTES = RUN / "NOTES.md"
BOARD = ROOT / "optimization" / "GPU_RUN_BOARD.md"

A_PEND = (b"`pending` means the chain has not reached the row yet; row 35 (`n1i8s3`) is the only `pending` "
          b"cell now, because `it36corrupt2` reaches it first (its earlier points "
          b"`l2i4s4`/`l3i4s4`/`l2i8s2` are gate-skipped - rows 23, 33 and 34 fail the loss gate).")
A_STAT = b"row 35 (`n1i8s3`) PASSES**"
A_NOTE = b"## Live status\n\n"
MARK = b"## Amendment 27"


def main() -> int:
    v = json.loads(VERD.read_text(encoding="utf-8"))
    t = v["totals"]
    low = t["lowrank_raw"]
    pid = v["tag"]

    # ---------------------------------------------------------------- corruption paragraph
    new_pend = (
        b"`pending` means the chain has not reached the row yet, and **since 2026-11-22 23:26 no cell is "
        b"`pending` any more**: `it36corrupt2` checked row 35 (`n1i8s3`) - the smallest passing row, "
        b"3,392 bits - and it PASSES (50/50 users, "
        + f"{low['calls']:,}".encode()
        + b" raw card/note state observations, 0 non-finite, max |state| "
        + f"{low['maxabs']:.4f}".encode()
        + b", 150 p(recall) tensors, 0 non-finite, "
        + f"{v['secs'] / 60.0:.1f}".encode()
        + b" min; its earlier points `l2i4s4`/`l3i4s4`/`l2i8s2` were gate-skipped because rows 23, 33 and "
        b"34 fail the loss gate, and its last four are `it36noterank2` rows already recorded). So every "
        b"row that passes the log-loss gate now also passes the finiteness gate, and the corruption "
        b"column has no open cell."
    )
    raw = RESULTS.read_bytes()
    if b"since 2026-11-22 23:26 no cell is" in raw:
        print("RESULTS: corruption paragraph already updated")
    else:
        if raw.count(A_PEND) != 1:
            raise SystemExit(f"ABORT: A_PEND found {raw.count(A_PEND)} times")
        RESULTS.write_bytes(raw.replace(A_PEND, new_pend, 1))
        print("RESULTS corruption paragraph: ok")

    # ---------------------------------------------------------------- status sentence
    raw = RESULTS.read_bytes()
    if b"PASSES all six constraints, corruption included" in raw:
        print("RESULTS: status sentence already updated")
    else:
        if raw.count(A_STAT) != 1:
            raise SystemExit(f"ABORT: A_STAT found {raw.count(A_STAT)} times")
        RESULTS.write_bytes(raw.replace(
            A_STAT, b"row 35 (`n1i8s3`) PASSES all six constraints, corruption included**", 1))
        print("RESULTS status sentence: ok")

    # ---------------------------------------------------------------- NOTES bullet
    bullet = (
        "* 2026-11-22 23:26 - **(row 35) `n1i8s3` PASSES the corruption gate, so the corruption column "
        "has no open cell and 3,392 bits is a fully deployable size.** `it36corrupt2` scored the 50-user "
        f"roster on `qat_it36n1i8s3/qat_124.pth`: 50/50 users, {low['calls']:,} raw card/note state "
        f"observations (every one BEFORE `_sanitize_state`), 0 NaN, 0 +/-inf, max |state| "
        f"{low['maxabs']:.4f} (quantized {t['lowrank_quant']['maxabs']:.4f}), 582 shift-vector passes "
        f"max |x| {t['shift_in']['maxabs']:.4f}, 150 p(recall) tensors and 0 non-finite, "
        f"{v['secs'] / 60.0:.1f} min. Verdict `corruption_corrupt_n1i8s3.json`; `record_corruption.py` "
        "then wrote the cell, so rows 15-18 and 29-32 keep their PASSes and row 35 stops being "
        "`pending`. With that, `it36corrupt2` logged `=== it36corrupt2 chain end ===` at 23:26:01 - every "
        "one of its remaining points was either gate-skipped or already recorded - and `it36e1` (the "
        "tail-energy penalty, E1) took the GPU at 23:29:46 with its fused-off control `l2i8s4pf`.\n"
    )
    nb = NOTES.read_bytes()
    if b"(row 35) `n1i8s3` PASSES the corruption gate" in nb:
        print("NOTES: bullet already present")
    else:
        if nb.count(A_NOTE) != 1:
            raise SystemExit(f"ABORT: A_NOTE found {nb.count(A_NOTE)} times")
        NOTES.write_bytes(nb.replace(A_NOTE, A_NOTE + bullet.encode("utf-8"), 1))
        print("NOTES bullet: ok")

    # ---------------------------------------------------------------- BOARD Amendment 27
    amend = (
        "\n\n## Amendment 27 (2026-11-22 23:26): row 35 PASSES the corruption gate - the corruption "
        "column is CLOSED and 3,392 bits is fully deployable\n\n"
        "`it36corrupt2` scored `n1i8s3` (card rank-2 int8 + int4 shifts, note rank-1 int8 + int3 shifts, "
        "1,280/2,112 = 3,392 bits) on the 50-user long-history roster: "
        f"{v['users_scored']}/{v['users_requested']} users, {low['calls']:,} raw card/note state "
        f"observations before `_sanitize_state`, 0 NaN, 0 +inf, 0 -inf, max |state| {low['maxabs']:.4f} "
        f"(after quantization {t['lowrank_quant']['maxabs']:.4f}), {t['shift_in']['calls']} shift-vector "
        f"passes at max |x| {t['shift_in']['maxabs']:.4f}, 150 p(recall) tensors with 0 non-finite, "
        f"{v['secs'] / 60.0:.1f} min on CUDA. **PASS** (`corruption_corrupt_n1i8s3.json`).\n\n"
        "This closes the corruption column: rows 15, 16, 17, 18, 29, 30, 31, 32 and 35 all PASS, every "
        "other row is a gate FAIL that is never checked, and no cell says `pending` any more. Row 35 is "
        "therefore the smallest recipe that satisfies ALL SIX constraints - the loss ceiling on both "
        "heads, both bit bars, the 0.75x speed floor (0.789x) and finiteness at 3,392 bits - and it is "
        "the number to beat.\n\n"
        "`it36corrupt2` then logged `=== it36corrupt2 chain end ===` at 23:26:01: its first three points "
        "are gate Fails (rows 23, 33, 34), rows 39-43 fail too, and its last four are `it36noterank2` "
        "rows already recorded. `it36e1` took the GPU at 23:29:46 with `l2i8s4pf`, its fused-off QAT "
        "control, followed by the two lambda brackets `l2i8s4e1a`/`l2i8s4e1b`.\n"
    )
    bb = BOARD.read_bytes()
    if MARK in bb:
        print("BOARD: Amendment 27 already present")
    else:
        BOARD.write_bytes(bb + amend.encode("utf-8"))
        print("BOARD: Amendment 27 appended")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())