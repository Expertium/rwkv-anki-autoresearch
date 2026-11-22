"""Add the fixed-basis (approach 1) candidates to quant_sweep.CANDIDATES.

Each candidate's QAT scope carries m (modes per side) and the coefficient int-N level; the third
tuple element is the explicit bits override, because `bits_for` only knows the rank-r and full-matrix
formulas. Bits per layer = m*m*qbits (the stored coefficient block) + 64*shift_bits (the token-shift
vectors, unchanged), and the note has 3 layers.

CRLF: quant_sweep.py is CRLF-dominant, so this patcher works on raw bytes and anchors on the
single-line `l2i4s4n4` entry (the same anchor the int6-split patcher used).

Usage: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_add_fixedbasis_candidates.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QS = ROOT / "scratchpad" / "quant_plan" / "quant_sweep.py"
BASIS = ROOT / "scratchpad" / "quant_plan" / "corruption" / "fixedbasis_l2i8s4all.npz"

ANCHOR = (b'                 "row 17 layout at int4 factors, the bottom rung (768/1,536 = 2,304 '
          b'bits)", None),\r\n')

LAYERS = {"card": 1, "note": 3}
SHIFT_BITS = 4          # card:int4,note:int4 in every fixed-basis candidate
QBITS = {"int8": 8, "int6": 6, "int5": 5, "int4": 4}

# (name, m, level)
CANDIDATES = [("fb4i8", 4, "int8"), ("fb6i8", 6, "int8"), ("fb8i8", 8, "int8"),
              ("fb12i8", 12, "int8"), ("fb8i4", 8, "int4"), ("fb16i8", 16, "int8")]


def bits(m, level):
    per_layer = m * m * QBITS[level] + 64 * SHIFT_BITS
    return {s: LAYERS[s] * per_layer for s in LAYERS}


def main() -> int:
    raw = QS.read_bytes()
    if b'"fb8i8"' in raw:
        print("quant_sweep.py already carries the fixed-basis candidates")
        return 0
    n = raw.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times, expected 1")

    new = b""
    basis_lit = str(BASIS).replace("\\", "\\\\")   # a Windows path needs doubled backslashes in a
    for name, m, level in CANDIDATES:              # Python string literal (\U in \Users is an escape)
        b = bits(m, level)
        new += (f'    "{name}": ({{"RWKV_QAT_LOWRANK_SCOPE": "card:{m}:{level},note:{m}:{level}",\r\n'
                f'               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",\r\n'
                f'               "RWKV_FIXEDBASIS_PATH": "{basis_lit}"}},\r\n'
                f'              "FIXED-BASIS state coding (approach 1): m={m} modes per side, '
                f'{level} coefficient block '
                f'({b["card"]}/{b["note"]} = {b["card"] + b["note"]} bits)",\r\n'
                f'              ({b["card"]}, {b["note"]})),\r\n').encode("utf-8")

    QS.write_bytes(raw.replace(ANCHOR, ANCHOR + new, 1))
    print(f"patched quant_sweep.py (+{len(new)} bytes)")
    for name, m, level in CANDIDATES:
        b = bits(m, level)
        print(f"  {name:>7}: m={m:<2} {level:>4}  card {b['card']:>5}  note {b['note']:>5}  "
              f"total {b['card'] + b['note']:>5}")
    print(f"basis path: {BASIS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
