"""Register the `l2i6s4n1i8s3` candidate: card rank-2 int6 + int4 shifts, note rank-1 int8 + int3 shifts.

WHY. Two measured prices meet at 3,136 bits. (a) `l2i6c` - the CARD at int6, note untouched - costs
only +0.000052 ahead / +0.000115 imm as a per-user PAIRED delta on 11 long histories
(`rust_wide_screen_it36int6split.json`), and it is DEPLOYABLE because `parse_level` already accepts
int6. (b) Row 35's note int3 SHIFT rung costs +0.000011/+0.000037 on the same kind of test for 192
fewer note bits. Stacking them gives card 1,024 + note 2,112 = 3,136 bits, i.e. 256 BELOW row 35
(3,392, the smallest passing row) and 448 below row 17. Row 41 (`l3i4s4n8s3`) already showed that
this SIZE is reachable - it missed the imm ceiling by 0.000016 - but it bought the card's 1,024 bits
with rank-3 int4 factors (+0.000472 imm alone), where int6 rank-2 is priced 4x cheaper.

Adds the candidate to `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`.
Raw-byte replaces: all three files mix CRLF and LF.

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_add_int6combo_candidate.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SWEEP = RUN / "quant_sweep.py"
GRID = RUN / "rust_grid.py"
CORRUPT = RUN / "corruption" / "check_corruption.py"

NAME = "l2i6s4n1i8s3"

# ---------------------------------------------------------------- quant_sweep.CANDIDATES
SWEEP_ANCHOR = (
    b'"l2i6n": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int6",\n'
    b'               "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\n'
    b'              "row 17 layout with the NOTE at int6 and the card left at int8 (1,280/1,920 = '
    b'3,200 bits)",\n'
    b'              None),\n'
)
SWEEP_NEW = (
    b'    "l2i6s4n1i8s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\n'
    b'                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},\n'
    b'                     "card rank-2 int6 + int4 shifts and note rank-1 int8 + int3 shifts "\n'
    b'                     "(1,024/2,112 = 3,136 = 256 BELOW row 35, the smallest passing row)",\n'
    b'                     (1024, 2112)),\n'
)

# ---------------------------------------------------------------- rust_grid.CONFIGS
GRID_ANCHOR = b"    # FIXED-BASIS (codec) state coding (approach 1, 2026-11-22)"
GRID_NEW = (
    b'    # The int6 CARD rung (+0.000052/+0.000115 on 11 long histories) stacked on the note\'s int3\n'
    b'    # SHIFT rung (row 35, +0.000011/+0.000037): 1,024 + 2,112 = 3,136 bits, 256 below row 35.\n'
    b'    "l2i6s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\n'
    b'                     "RWKV_QUANT_SHIFTS": "1",\n'
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},\n'
    b"\n"
)

# ---------------------------------------------------------------- check_corruption.CONFIGS
COR_ANCHOR = (
    b'    "l2i8fb6n": (RUN / "qat_it36l2i8fb6n" / "qat_124.pth", "l2i8fb6n",\n'
    b'                 "chain 19 point 2 (fixed-basis note m=6, 672 bits below row 17)"),\n'
    b'}\n'
)
COR_NEW = (
    b'    "l2i8fb6n": (RUN / "qat_it36l2i8fb6n" / "qat_124.pth", "l2i8fb6n",\n'
    b'                 "chain 19 point 2 (fixed-basis note m=6, 672 bits below row 17)"),\n'
    b'    "l2i6s4n1i8s3": (RUN / "qat_it36l2i6s4n1i8s3" / "qat_124.pth", "l2i6s4n1i8s3",\n'
    b'                     "chain 21 point 1 (int6 card + note int3 shifts, 3,136 bits)"),\n'
    b'}\n'
)


def _nl(raw: bytes) -> bytes:
    """The file's dominant line ending, so every anchor can be written with plain \\n."""
    return b"\r\n" if raw.count(b"\r\n") * 2 > raw.count(b"\n") else b"\n"


def patch(path: Path, old: bytes, new: bytes, label: str) -> int:
    raw = path.read_bytes()
    nl = _nl(raw)
    if nl != b"\n":
        old = old.replace(b"\r\n", b"\n").replace(b"\n", nl)
        new = new.replace(b"\r\n", b"\n").replace(b"\n", nl)
    if NAME.encode() in raw and old not in raw:
        print(f"{path.name}: already registered")
        return 0
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times in {path.name}, expected 1")
    path.write_bytes(raw.replace(old, new, 1))
    print(f"{path.name}: {label} added")
    return 1


def main() -> int:
    changed = 0
    changed += patch(SWEEP, SWEEP_ANCHOR, SWEEP_ANCHOR + SWEEP_NEW, "CANDIDATES entry")
    changed += patch(GRID, GRID_ANCHOR, GRID_NEW + GRID_ANCHOR, "CONFIGS entry")
    changed += patch(CORRUPT, COR_ANCHOR, COR_NEW, "CONFIGS entry")
    print(f"done ({changed} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())