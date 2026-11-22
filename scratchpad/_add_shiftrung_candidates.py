"""Register the three SHIFT-width rungs as QAT candidates in `quant_sweep.CANDIDATES`.

WHY. Once int6 turned out to be the FACTOR knee on both streams (rows 46/47 fail, the `it36wide6`
screen agrees), the only measured-cheap knob left below row 45's 2,944 bits is the SHIFT width: row 35
measured the note's int4 -> int3 step at +0.000011/+0.000037 for 192 bits. These three candidates spend
exactly that knob on row 45's layout, from the least to the most aggressive:

    l2i6s3n1i6s4   card 960 / note 1,920 = 2,880   card shifts int4 -> int3 only
    l2i6s3n1i6s3   card 960 / note 1,728 = 2,688   both streams' shifts at int3
    l2i6s2n1i6s3   card 896 / note 1,728 = 2,624   card shifts int3 -> int2 as well

Bit math (card = 2x2x32xq + 64xs, note = 3x(2x1x32xq + 64xs), q = factor bits, s = shift bits):
768+64s / 3x(384+64s). `rust_grid.py --tag it36wide7` prices the same three layouts on the CPU; the QAT
run is what decides the gate, because the screen is PTQ.

Line endings: quant_sweep.py is CRLF-dominant with 0 bare LF, so the edit is done in LF and written back
as CRLF with a byte round-trip check.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scratchpad" / "quant_plan" / "quant_sweep.py"

ANCHOR = ("    # THE NOTE-ONLY FIXED-BASIS CANDIDATES (2026-11-22, after the leave-one-user-out "
          "measurement).")

BLOCK = '''    # THE SHIFT-WIDTH RUNGS (2026-11-23): row 35 measured the note's int4 -> int3 SHIFT step at only
    # +0.000011/+0.000037 for 192 bits, and int6 is now the FACTOR knee on both streams, so shift width
    # is the only measured-cheap knob left below row 45's 2,944 bits. A CPU deploy screen
    # (`rust_grid.py --tag it36wide7`) prices the same three layouts; the QAT run is what decides the
    # gate, because the screen is PTQ against weights trained for int8 factors.
    # Bit math: card 2x2x32xq + 64xs = 768+64s, note 3x(2x1x32xq + 64xs) = 3x(384+64s).
    #   l2i6s3n1i6s4   card  960  note 1,920 = 2,880   (card shifts int4 -> int3 only)
    #   l2i6s3n1i6s3   card  960  note 1,728 = 2,688   (both streams' shifts at int3)
    #   l2i6s2n1i6s3   card  896  note 1,728 = 2,624   (card shifts int3 -> int2 as well)
    "l2i6s3n1i6s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int4"},
                     "row 45 with the CARD's shifts int4 -> int3 (960/1,920 = 2,880 bits)", None),
    "l2i6s3n1i6s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int3,note:int3"},
                     "row 45 with BOTH streams' shifts int4 -> int3 (960/1,728 = 2,688 bits)", None),
    "l2i6s2n1i6s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int2,note:int3"},
                     "row 45 with the card's shifts int2 and the note's int3 (896/1,728 = 2,624 bits)",
                     None),
'''

NAMES = ('"l2i6s3n1i6s4"', '"l2i6s3n1i6s3"', '"l2i6s2n1i6s3"')


def main() -> int:
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")
    print(f"line endings: CRLF {text.count(chr(13) + chr(10))}, bare LF "
          f"{text.count(chr(10)) - text.count(chr(13) + chr(10))}")

    lf = text.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    for name in NAMES:
        if name in lf:
            print(f"FAIL: {name} already registered")
            return 1

    out_lf = lf.replace(ANCHOR, BLOCK + ANCHOR, 1)
    for name in NAMES:
        # each candidate name appears exactly ONCE: as the dict key (no checkpoint path mentions it)
        n = out_lf.count(name)
        if n != 1:
            print(f"FAIL: bad insert for {name}: {n} occurrence(s)")
            return 1
    if out_lf.count("THE SHIFT-WIDTH RUNGS") != 1:
        print("FAIL: new block missing or duplicated")
        return 1

    out = out_lf.replace("\n", "\r\n")
    TARGET.write_bytes(out.encode("utf-8"))
    if TARGET.read_bytes().decode("utf-8") != out:
        print("FAIL: round trip differs")
        return 1
    print(f"ok: {len(raw)} -> {len(out)} bytes; all three rungs registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
