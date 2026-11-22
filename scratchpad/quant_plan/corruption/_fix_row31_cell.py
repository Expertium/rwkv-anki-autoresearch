"""Refresh the two stale "the deploy engine cannot express a per-stream shift level" claims.

Both RESULTS.md row 31's cell and the recording script that wrote it say the row is LOSS-ONLY because
`RWKV_STATE_SHIFT_LEVEL` took one bare level. That was fixed on 2026-11-22 (`CompressCfg::shift_level`,
GPU_RUN_BOARD Amendment 23), so both files are corrected in place with raw-byte replaces (both mix
CRLF and LF, so the text editor cannot match them).

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_fix_row31_cell.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"
SCRIPT = ROOT / "scratchpad" / "quant_plan" / "corruption" / "_record_l2i8n2i8s8.py"

OLD = (
    b"**LOSS-ONLY: the deploy engine cannot express a per-stream "
    b"shift level.** `RWKV_STATE_SHIFT_LEVEL` takes ONE bare level, and a per-stream string such as "
    b"`card:int4,note:int8` is silently dropped (`parse_level` in `rust/rwkv-infer/src/model.rs` "
    b"matches only `intN`), after which the shifts fall back to the WKV FACTOR level. Verified "
    b"2026-11-22 by running the engine three times on user 107: `card:int4,note:int8` gave predictions "
    b"identical to `int8` (max abs diff 0.0)"
)
NEW = (
    b"**Deployable on the speed axis since 2026-11-22**: "
    b"`RWKV_STATE_SHIFT_LEVEL` is now a PER-STREAM map (`CompressCfg::shift_level` in "
    b"`rust/rwkv-infer/src/model.rs`, GPU_RUN_BOARD Amendment 23), so `card:int4,note:int8` is "
    b"expressible; when this row was recorded the option took ONE bare level and a per-stream string "
    b"was silently dropped, after which the shifts followed the WKV FACTOR level. What is still "
    b"missing is its deploy SPEED: the rank-2 note path needs `RWKV_LOWRANK_FAST=1`, which reads "
    b"0.803x on the neighbouring `l2i8n2i8` config, and at 5,888 bits the row is a loss instrument "
    b"rather than an axis-1 candidate"
)


def patch(path: Path) -> int:
    raw = path.read_bytes()
    n = raw.count(OLD)
    if n == 0:
        if NEW in raw:
            print(f"{path.name}: already corrected")
            return 0
        print(f"{path.name}: anchor absent (already reworded elsewhere) - skipped")
        return 0
    path.write_bytes(raw.replace(OLD, NEW))
    print(f"{path.name}: corrected ({n} site(s))")
    return 1


def main() -> int:
    changed = patch(RESULTS) + patch(SCRIPT)
    print(f"done ({changed} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
