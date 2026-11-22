"""Register chain 18's two MIXED-precision int6 points in `check_corruption.CONFIGS`.

`l2i6c` = card rank-2 int6 + note rank-1 int8 (1,024/2,304 = 3,328 bits) and `l2i6n` = card rank-2
int8 + note rank-1 int6 (1,280/1,920 = 3,200 bits) are the two points of chain 18 (`it36int6card`).
Both sit ABOVE row 45's 2,944 bits, so they are diagnostic rows - they say WHICH stream owns the int6
tax - but a row that passes the log-loss gate still needs its own finiteness check, per Andrew's
standing order.

The second tuple field is the `quant_sweep.CANDIDATES` key that supplies the deploy env, exactly like
every other entry, so the env is whatever the QAT run itself used.

Line endings: check_corruption.py is CRLF-dominant with 0 bare LF, so the edit is done in LF and the
file is written back as CRLF, with a byte round-trip check at the end.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scratchpad" / "quant_plan" / "corruption" / "check_corruption.py"

ANCHOR = "    # CHAIN 24: the E1 tail penalty on row 45's frontier layout"

BLOCK = '''    # CHAIN 18 (it36int6card, 2026-11-23): the MIXED-precision int6 points, which say WHERE the
    # int6 tax comes from. Both sit ABOVE row 45's 2,944 bits (3,328 and 3,200), so they are
    # diagnostic rows rather than frontier rows - but each needs its finiteness check like any row.
    "l2i6c": (RUN / "qat_it36l2i6c" / "qat_124.pth", "l2i6c",
              "chain 18 point 1 (card rank-2 int6 + note rank-1 int8, 3,328 bits)"),
    "l2i6n": (RUN / "qat_it36l2i6n" / "qat_124.pth", "l2i6n",
              "chain 18 point 2 (card rank-2 int8 + note rank-1 int6, 3,200 bits)"),
'''


def main() -> int:
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")
    print(f"line endings: CRLF {text.count(chr(13) + chr(10))}, bare LF "
          f"{text.count(chr(10)) - text.count(chr(13) + chr(10))}")

    lf = text.replace("\r\n", "\n")
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    if '"l2i6c"' in lf or '"l2i6n"' in lf:
        print("FAIL: l2i6c/l2i6n already registered")
        return 1

    out_lf = lf.replace(ANCHOR, BLOCK + ANCHOR, 1)
    for name in ('"l2i6c"', '"l2i6n"'):
        # the quoted name appears TWICE per entry: the dict key and the CANDIDATES key of the
        # second tuple field (which supplies the deploy env). Not 3x: the checkpoint path is
        # "qat_it36l2i6c", an unquoted-adjacent string, and it does not match the quoted pattern.
        n = out_lf.count(name)
        if n != 2:
            print(f"FAIL: bad insert for {name}: {n} occurrence(s)")
            return 1
    if out_lf.count("CHAIN 24: the E1 tail penalty") != 1:
        print("FAIL: CHAIN 24 comment lost or duplicated")
        return 1
    if out_lf.count("CHAIN 18 (it36int6card") != 1:
        print("FAIL: new block missing")
        return 1

    out = out_lf.replace("\n", "\r\n")
    TARGET.write_bytes(out.encode("utf-8"))
    back = TARGET.read_bytes().decode("utf-8")
    if back != out:
        print("FAIL: round trip differs")
        return 1
    print(f"ok: {len(raw)} -> {len(out)} bytes; CONFIGS entries: "
          f"{out_lf.count('RUN / \"qat_')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
