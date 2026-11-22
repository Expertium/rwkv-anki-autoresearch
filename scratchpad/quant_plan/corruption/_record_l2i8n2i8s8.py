"""Record chain 14 point 3 (`l2i8n2i8s8`) as RESULTS row 31, plus the three bundle findings.

Both files mix CRLF and LF, so every anchor is matched as RAW BYTES and asserted unique.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
NOTES = ROOT / "scratchpad" / "quant_plan" / "NOTES.md"
RECORDED = ROOT / "scratchpad" / "quant_plan" / "autorun" / "RECORDED.txt"

ROW31 = (
    "| 31 | QAT card rank-2 int8 + int4 shifts (row 17's card), note **rank-2 int8 + int8 shifts** "
    "- is the note's SHIFT precision free at rank 2? (`l2i8n2i8s8`) | 1,280 (160) | 4,608 (576) | "
    "n/a | n/a | 0.333683 | 0.306295 | -0.000119 | +0.000084 | **PASS on both heads, and the "
    "lowest imm tax of the note-rank-2 family**: ahead is BETTER than the anchor by 0.000119 "
    "(ub95 -0.000088, p_worse 1.0000) and imm is 0.000084 above it (ub95 +0.000108, p_worse 7.2e-11) "
    "- but it spends 768 MORE note bits than row 29 (4,608 vs 3,840) to buy 0.000019 of imm, so an "
    "int8 note shift level is not free. **Deployable on the speed axis since 2026-11-22**: "
    "`RWKV_STATE_SHIFT_LEVEL` is now a PER-STREAM map (`CompressCfg::shift_level` in "
    "`rust/rwkv-infer/src/model.rs`, GPU_RUN_BOARD Amendment 23), so `card:int4,note:int8` is "
    "expressible; when this row was recorded the option took ONE bare level and a per-stream string "
    "was dropped, after which the shifts followed the WKV FACTOR level. The deploy SPEED is still "
    "missing: the rank-2 note path needs `RWKV_LOWRANK_FAST=1` (0.803x on `l2i8n2i8`), and at 5,888 "
    "bits the row is a loss instrument rather than an axis-1 candidate | pending |\n"
)

PARA = (
    "**Row 31 (`l2i8n2i8s8`, chain 14 point 3) passes both heads, and settles that the note's SHIFT "
    "precision is not free.** Card rank-2 int8 + int4 shifts (row 17's card) with the note at rank-2 "
    "int8 + int8 shifts: 1,280/4,608 = 5,888 bits, ahead 0.333683 (delta -0.000119, ub95 -0.000088) "
    "and imm 0.306295 (delta +0.000084, ub95 +0.000108). It is the best imm tax of the note-rank-2 "
    "family (row 29 +0.000103, row 30 -0.000014 at rank 3 on the card), but 768 extra note bits buy "
    "only 0.000019 of imm. **The row is loss-only until the engine's shift-level parser gains "
    "per-stream support**: `RWKV_STATE_SHIFT_LEVEL` accepts one bare `intN`, and a per-stream string "
    "is silently ignored (the shifts then follow the WKV factor level). That silent fallback also "
    "applies to any earlier recorded row whose config used a per-stream shift string; `c2_nl1i8` is "
    "unaffected, because its factor levels (card int2, note int8) are exactly the shift levels it "
    "intended."
)

BULLET = (
    "* 2026-11-22 15:25 - **Chain 14 point 3 lands as RESULTS row 31 (`l2i8n2i8s8`, card rank-2 int8 "
    "+ int4 shifts, note rank-2 int8 + int8 shifts, 1,280/4,608 = 5,888 bits): PASS on both heads** "
    "- 0.333683/0.306295 = -0.000119 ahead / +0.000084 imm vs `qnj_it36ctl` (ub95 -0.000088/"
    "+0.000108, p_worse 1.0000/7.2e-11). The best imm tax of the note-rank-2 family, but it spends "
    "768 more note bits than row 29 for 0.000019 of imm, so int8 note shifts are not free. "
    "**Loss-only: `RWKV_STATE_SHIFT_LEVEL` takes ONE bare level** and `card:int4,note:int8` is "
    "silently ignored (verified: predictions identical to `int8`, 0.087 away from `int4`), so a "
    "per-stream shift level needs a parser change before this row can be quoted against constraint 6. "
    "Chain 14 moved to its last point (`l2i8n3i8`) at 14:56:47.\n"
    "\n"
    "* 2026-11-22 15:30 - **Three findings from building the iter-17 bundle, recorded because they "
    "change how earlier numbers must be read.** (1) `reference/trace_user_*` is a June-2026 export "
    "whose FEATURE recipe differs from the current one (same review indices, feature values up to 3.0 "
    "apart), so scoring current weights on it is out of distribution: the deploy tax measured there "
    "read +0.0027 imm / -0.0030 ahead, while the same measurement on a fresh current-lineage export "
    "reads +0.000316 imm / +0.000335 ahead, in line with row 17's own 750-user gate (+0.000327 imm). "
    "`rust_score.json`, `rust_speed_idle.json` and `rust_finite_probe.json` stay valid as INTERNAL "
    "comparisons (Rust vs Rust on one trace set) but not as levels. (2) The Python RNN inference path "
    "(`rwkv_rnn_model` -> `rwkv_ops.single_timestep`) does NOT apply the state quantization; only the "
    "training/eval path (`quant_aware_rwkv7`) does. A trace's embedded `py_pred_*` is therefore an "
    "UNQUANTIZED reference: the engine's `plain` config matches it to <=8.5e-07 per review and 0.000000 "
    "log loss on 3,465 / 3,865 / 12,915 / 23,455-review histories, and a quantized Python<->Rust "
    "comparison must come from the eval path. (3) The AHEAD head is not comparable between the eval "
    "path (LMDB) and the trace path: they differ by about 0.005 log loss on a long user WITH and "
    "WITHOUT quantization (the unquantized control shows the same gap), i.e. it is an input difference "
    "in the exporter's frame replica, not a quantization difference. The IMM head agrees to 8e-05 on "
    "the long user. Compare ahead numbers only within one path.\n"
)


def sub(raw: bytes, old: bytes, new: bytes, label: str) -> bytes:
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    print(f"  {label}: ok")
    return raw.replace(old, new, 1)


def main() -> int:
    res = RESULTS.read_bytes()
    if b"| 31 |" in res:
        # This script is ONE-SHOT: it appends row 31 after row 30's cell, so running it twice would
        # write a duplicate row. Row 31 is already in RESULTS.md (and was later corrected in place by
        # _fix_row31_cell.py when the per-stream shift level landed).
        print("  RESULTS: row 31 already recorded; nothing to do")
        return 0
    anchor = b"rust_speed_fastrank_on.json`; 0.611x without it) | pending |"
    res = sub(res, anchor, anchor + b"\n" + ROW31.encode("utf-8").rstrip(b"\n"), "RESULTS row 31")
    res = sub(res, b"**Row 19 (`nr4`, the note-pool cut) FAILS**",
              PARA.encode("utf-8") + b"\n\n**Row 19 (`nr4`, the note-pool cut) FAILS**",
              "RESULTS row 31 paragraph")
    RESULTS.write_bytes(res)

    notes = NOTES.read_bytes()
    notes = sub(notes, b"## Live status\n\n", b"## Live status\n\n" + BULLET.encode("utf-8"),
                "NOTES bullets")
    NOTES.write_bytes(notes)

    rec = RECORDED.read_text(encoding="utf-8")
    if "it36l2i8n2i8s8" in rec:
        print("  RECORDED.txt: already stamped")
    else:
        RECORDED.write_text(rec.rstrip("\n") + "\nit36l2i8n2i8s8\n", encoding="utf-8")
        print("  RECORDED.txt: stamped it36l2i8n2i8s8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
