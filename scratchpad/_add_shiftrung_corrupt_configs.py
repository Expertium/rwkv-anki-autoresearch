"""Register the three SHIFT-width rungs in `check_corruption.CONFIGS`.

Companion to `scratchpad/_add_shiftrung_candidates.py`. Andrew's standing order is that a row which
passes the log-loss gate gets its finiteness check immediately, so the rung that wins has to be
runnable the moment its sweep lands:

    l2i6s3n1i6s4   card  960 / note 1,920 = 2,880   (card shifts int4 -> int3)
    l2i6s3n1i6s3   card  960 / note 1,728 = 2,688   (both streams' shifts at int3)
    l2i6s2n1i6s3   card  896 / note 1,728 = 2,624   (card shifts int2 as well)

The second tuple field is the `quant_sweep.CANDIDATES` key that supplies the deploy env - the same name,
because the rung's candidate env IS the deploy env.

Line endings: check_corruption.py is CRLF-dominant with 0 bare LF; the edit is done in LF and written
back as CRLF with a byte round-trip check.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = "    # CHAIN 18 (it36int6card, 2026-11-23): the MIXED-precision int6 points"

BLOCK = '''    # THE SHIFT-WIDTH RUNGS (2026-11-23): shift width is the only measured-cheap knob left below
    # row 45's 2,944 bits. The winner of the pair's own sweep gets its finiteness check here.
    "l2i6s3n1i6s4": (RUN / "qat_it36l2i6s3n1i6s4" / "qat_124.pth", "l2i6s3n1i6s4",
                     "shift rung 1 (card shifts int4 -> int3, 960/1,920 = 2,880 bits)"),
    "l2i6s3n1i6s3": (RUN / "qat_it36l2i6s3n1i6s3" / "qat_124.pth", "l2i6s3n1i6s3",
                     "shift rung 2 (both streams' shifts int3, 960/1,728 = 2,688 bits)"),
    "l2i6s2n1i6s3": (RUN / "qat_it36l2i6s2n1i6s3" / "qat_124.pth", "l2i6s2n1i6s3",
                     "shift rung 3 (card shifts int2, note int3, 896/1,728 = 2,624 bits)"),
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
        # twice per entry: the dict key AND the candidate key of the second tuple field
        n = out_lf.count(name)
        if n != 2:
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
