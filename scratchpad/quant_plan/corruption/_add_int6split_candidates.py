"""Add the two MIXED-precision int6 candidates to quant_sweep.CANDIDATES.

The existing int6/int5 ladder coarsens card and note together, so a result cannot say WHERE the int6
tax comes from. An 11-user deploy screen (rust_wide_screen_it36int6split.json) says it is mostly the
NOTE's: card-only int6 costs +0.000115 imm / +0.000052 ahead, note-only int6 costs +0.000307 imm /
+0.000268 ahead (paired vs l2i8s4). The card-only rung is therefore the one worth a 750-user QAT point.

Raw bytes. The anchor is the single line that ends the `l2i4s4n4` entry (CRLF file), which is unique.
"""
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
SWEEP = RUN / "quant_sweep.py"

ANCHOR = (b'                 "row 17 layout at int4 factors, the bottom rung (768/1,536 = 2,304 bits)", '
          b'None),\r\n')

NEW = (
    b'\r\n'
    b'    # MIXED precision (2026-11-22 20:20): the ladder above coarsens BOTH streams at once, so a'
    b' result cannot say\r\n'
    b'    # WHERE the int6 tax comes from. An 11-user deploy screen says it is mostly the NOTE\'s: card'
    b' int6 + note\r\n'
    b'    # int8 (`l2i6c`) costs +0.000115 imm [+0.000018,+0.000218] and +0.000052 ahead'
    b' [-0.000007,+0.000109], while\r\n'
    b'    # card int8 + note int6 (`l2i6n`) costs +0.000307 imm [+0.000124,+0.000558] and +0.000268'
    b' ahead\r\n'
    b'    # [+0.000079,+0.000601] (paired vs l2i8s4, rust_wide_screen_it36int6split.json). So the'
    b' CARD\'s int6 is\r\n'
    b'    # nearly free at PTQ and is the one worth a 750-user QAT point.\r\n'
    b'    "l2i6c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\r\n'
    b'               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    b'              "row 17 layout with the CARD at int6 and the note left at int8 (1,024/2,304 = 3,328'
    b' bits)",\r\n'
    b'              None),\r\n'
    b'    "l2i6n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int6",\r\n'
    b'               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    b'              "row 17 layout with the NOTE at int6 and the card left at int8 (1,280/1,920 = 3,200'
    b' bits)",\r\n'
    b'              None),\r\n'
)


def main() -> int:
    raw = SWEEP.read_bytes()
    if b'"l2i6c"' in raw:
        print("already present")
        return 0
    n = raw.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
    SWEEP.write_bytes(raw.replace(ANCHOR, ANCHOR + NEW, 1))
    print("patched quant_sweep.py (+%d bytes)" % len(NEW))
    return 0


if __name__ == "__main__":
    sys.exit(main())
