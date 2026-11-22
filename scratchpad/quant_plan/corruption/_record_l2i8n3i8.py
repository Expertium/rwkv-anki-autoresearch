"""Record chain 14 point 4 (`l2i8n3i8`) as RESULTS row 32: the note's rank ladder closes at rank 2.

Both RESULTS.md and NOTES.md mix CRLF and LF, so every anchor is matched as RAW BYTES and asserted
unique before anything is written.

The numbers come from `scratchpad/quant_plan/quant_sweep_it36l2i8n3i8.json` (landed 2026-11-22 16:25,
750 users, ref `qnj_it36ctl`, 3,362.6 s):
    ahead 0.33381661733333334  delta +1.4806666666664558e-05  ub95 +3.263212719970031e-05  p 0.2151
    imm   0.3063337626666666   delta +1.2236533333333306e-04  ub95 +1.4152972328071122e-04 p 2.0e-30
    bits  card 1,280  note 5,376  (total 6,656)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
RECORDED = ROOT / "scratchpad" / "quant_plan" / "autorun" / "RECORDED.txt"
STAMP = "it36l2i8n3i8"

ROW32 = (
    "| 32 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-3 int8** + int4 shifts "
    "- the note's rank ladder, last rung (`l2i8n3i8`) | 1,280 (160) | 5,376 (672) | n/a | n/a | "
    "0.333817 | 0.306334 | +0.000015 | +0.000122 | **PASS on both heads, and it CLOSES the note's "
    "rank ladder**: against row 29 (`l2i8n2i8`, the same card) rank 3 is WORSE on BOTH heads "
    "(-0.000115 ahead / +0.000103 imm at rank 2) while spending 1,536 MORE note bits (5,376 vs "
    "3,840), and against row 30 (rank 2 on a rank-3 card) it is worse still. The note's residual tax "
    "is therefore NOT rank-limited - extra note rank cannot buy it back - so the zero-bit levers "
    "(E1 tail penalty, chains 15/16) are the right next move rather than more note rank. Speed n/a: "
    "a rank-3 note needs `RWKV_LOWRANK_FAST=1` (0.791x at rank 2 on a rank-3 card, row 30), not yet "
    "measured for this exact pair | pending |"
)

PARA = (
    "**Row 32 (`l2i8n3i8`, chain 14 point 4) passes both heads and closes the note's rank ladder at "
    "rank 2.** Card rank-2 int8 + int4 shifts (row 17's card) with the note at rank-3 int8 + int4 "
    "shifts: 1,280/5,376 = 6,656 bits, ahead 0.333817 (delta +0.000015, ub95 +0.000033, p_worse "
    "0.215) and imm 0.306334 (delta +0.000122, ub95 +0.000142, p_worse 2.0e-30). The decisive "
    "comparison is row 29 `l2i8n2i8`, the same card with the note at rank 2 (3,840 note bits): it "
    "reads -0.000115 ahead / +0.000103 imm, i.e. BETTER on both heads for 1,536 fewer note bits. "
    "Note rank 3 also loses to rank 2 on a rank-3 card (row 30 -0.000144/-0.000014). So on the note, "
    "rank 2 dominates rank 3, and the note's remaining rank-1 tax is not a rank-truncation problem "
    "that more rank can pay off. That is exactly the premise E1 rests on: measure the tail energy the "
    "truncation discards and penalize it, at zero bits and zero deploy cost."
)

BULLET = (
    "* 2026-11-22 16:25 - **Chain 14 finished; its last point lands as RESULTS row 32 (`l2i8n3i8`, "
    "card rank-2 int8 + int4 shifts, note rank-3 int8 + int4 shifts, 1,280/5,376 = 6,656 bits): PASS "
    "on both heads, 0.333817/0.306334 = +0.000015 ahead / +0.000122 imm vs `qnj_it36ctl` (ub95 "
    "+0.000033/+0.000142).** It CLOSES the note's rank ladder: rank 3 is worse than rank 2 on BOTH "
    "heads (row 29 on the same card reads -0.000115 ahead / +0.000103 imm) for 1,536 MORE note bits, "
    "and worse than rank 2 on a rank-3 card too (row 30). The note's residual tax is not "
    "rank-limited, which is the premise the E1 tail-energy penalty rests on - chains 15/16 stay the "
    "right next move. `it36shrink` then took the GPU at 16:46 (candidate `l3i4s4`) after the queue's "
    "first free handoff; `it36equalbits` and `it36shrink` both waited only on an idle GPU, which is "
    "by design (`it36equalbits` waits on the old `it36corrupt` log, already ended).\n"
    "\n"
    "* 2026-11-22 16:45 - **`int6` and `int5` are now trainable, and the factor-precision ladder is "
    "queued as chain 17 (`it36int6`).** `rwkv/architecture.py:151` `_QMAX` held only "
    "int8/int4/int3/int2/fp32, so `card:2:int6` raised `KeyError` in `_apply_qat_scopes`; the Rust "
    "deploy engine has always spoken both levels (`parse_level`, `model.rs:2104-2105`) and both "
    "quantization paths are generic in qmax, so the keys were added (additive only - no existing "
    "recipe names them), together with `quant_sweep._QBITS`/`_QMAX` and three candidates at row 17's "
    "EXACT layout (card rank 2, note rank 1, int4 shifts): `l2i6s4` 1,024/1,920 = 2,944 bits (640 "
    "below row 17), `l2i5s4` 896/1,728 = 2,624 (960 below, and 192 below row 21), `l2i4s4n4` "
    "768/1,536 = 2,304 (1,280 below; the only BOTH-int4 combination - row 21 was note-int4 only and "
    "row 23 card-int4 only). A 2-user Rust PTQ probe on the row-17 checkpoint put `l2i6s4` at "
    "+0.000003 ahead / +0.000002 imm against `l2i8s4` for 640 fewer bits and `l2i5s4` at "
    "+0.000017/-0.000073, but the probe CANNOT settle the gate: calibrated against rows 21 and 23 it "
    "reproduces the note-int8->int4 cost at only 54% and gets the card-int8->int4 sign WRONG, so "
    "every point is a full 750-user QAT run. The int6 path was pre-flighted in the real trainer on "
    "CPU while the GPU was busy: the `[QAT-LOWRANK]` banner reported `card_id=rank2/fq31.0, "
    "note_id=rank1/fq31.0` and the first step trained finite (0.837127 / 0.3954 / 0.442); that "
    "pre-flight was then stopped, because any process matching the chains' BUSY_PAT strands the "
    "queue's handoffs.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    res = RESULTS.read_bytes()
    anchor = b"and 0.087 away from `int4` | pending |\n"
    res = sub(res, anchor, anchor + ROW32.encode("utf-8") + b"\n", "RESULTS row 32")
    res = sub(res, b"**Row 19 (`nr4`, the note-pool cut) FAILS**",
              PARA.encode("utf-8") + b"\n\n**Row 19 (`nr4`, the note-pool cut) FAILS**",
              "RESULTS row 32 paragraph")
    RESULTS.write_bytes(res)

    notes = NOTES.read_bytes()
    notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                "NOTES bullets")
    NOTES.write_bytes(notes)

    rec = RECORDED.read_text(encoding="utf-8")
    if STAMP in rec:
        print(f"  RECORDED.txt: {STAMP} already stamped")
    else:
        RECORDED.write_text(rec.rstrip("\n") + f"\n{STAMP}\n", encoding="utf-8")
        print(f"  RECORDED.txt: stamped {STAMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
