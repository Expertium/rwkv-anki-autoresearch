"""Insert the error-feedback closure paragraph into RESULTS.md (raw bytes, mixed newlines).

Anchors on the last sentence of the row-32 paragraph, which is unique in the file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "scratchpad" / "quant_plan" / "RESULTS.md"

ANCHOR = (b"measure the tail energy the truncation discards and penalize it, at zero bits and "
          b"zero deploy cost.")

PARA = (
    "\n\n**The error-feedback screen closes: EF is real but it cannot shrink the state.** "
    "`RWKV_LOWRANK_EF` is implemented in the deploy engine and had never been measured anywhere in this "
    "lineage. It is now screened on 21 of the longest corrupt50 histories, as per-user PAIRED deltas "
    "against `l2i8s4` inside each batch with 2,000 bootstrap resamples "
    "(`rust_wide_screen_ef_b3_ef_b2_ef_b1_fast_wide3_ef3.json`, instrument "
    "`corruption/wide_screen.py`). Three readings. (1) **The mechanism works**: at row 17's own config "
    "plus EF (`l2i8s4ef`, 1,536/3,072 = 4,608 bits) both heads improve with intervals clear of zero, "
    "-0.001167 ahead [-0.002081, -0.000339] and -0.002453 imm [-0.004826, -0.000539], better on 10/15 "
    "and 8/15 users. But the error buffer is real state, +1,024 bits, so that is axis 2 - and axis 2 is "
    "saturated (rows 29-31 sit at or below their own fp32 twins). (2) **EF cannot pay for itself on axis "
    "1**: the best layout is `l2i4s4n4ef` (both streams int4 + int4 buffer) at 1,024/2,304 = 3,328 bits, "
    "256 BELOW row 17, and it reads STATISTICALLY EQUAL to row 17 on both heads (+0.000049 ahead "
    "[-0.000737, +0.000875], -0.000376 imm [-0.001852, +0.000911], better on 10/21 and 6/21). A 3-user "
    "read of the same config had shown -0.001310 imm and looked like a 256-bit win; 21 users cut it to "
    "-0.000376 and an interval over zero. So EF at int4 factors buys a real but small -7% state at equal "
    "accuracy, no more. (3) **Coarsening the buffer removes the benefit**, because the buffer IS the "
    "mechanism: `l2i4s4n4ef2` (buffer at int2, 896/1,920 = 2,816 bits, exactly row 21's size) is clearly "
    "worse, +0.003439 ahead [+0.000587, +0.007513] and +0.002696 imm [+0.001449, +0.004236], better on "
    "1/6 and 0/6. EF is therefore closed as an axis-1 lever. It also cannot be gated today - "
    "`RWKV_LOWRANK_EF` exists only in the Rust engine while the gate is an eval-path measurement - so it "
    "needs no decision from Andrew unless a future recipe wants its -0.0025 imm at more bits.\n"
)


def main() -> int:
    raw = RESULTS.read_bytes()
    n = raw.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
    if b"The error-feedback screen closes" in raw:
        print("RESULTS: paragraph already present")
        return 0
    RESULTS.write_bytes(raw.replace(ANCHOR, ANCHOR + PARA.encode("utf-8"), 1))
    print("RESULTS paragraph: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
