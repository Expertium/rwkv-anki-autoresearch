"""Register the next rung down: `l2i6s4n1i8s2` - card rank-2 int6 + int4 shifts, note rank-1 int8 + int2 shifts.

WHY. The 3,136-bit point (`l2i6s4n1i8s3`) prices at +0.000164 ahead / +0.000205 imm on 7 long
histories (deploy-side, `rust_wide_screen_it36fbwide_b1_it36int6shift.json`), i.e. inside the ceiling
with the imm interval containing zero. The remaining lever on the note is its third-to-second SHIFT
bit: at int2 the note costs 1,536 + 3*64*2 = 1,920 bits instead of 2,112, taking the total to 2,944 -
192 below 3,136 and 448 below row 35 (3,392). Row 20 priced that rung at +0.000408 imm on an int4-
factor note, which is where the composition risk sits, so the point is registered as a GATED second
point of chain 21: it runs only if the 3,136-bit point PASSES the log-loss gate.

Adds the candidate to `quant_sweep.CANDIDATES`, `rust_grid.CONFIGS` and `check_corruption.CONFIGS`.
Raw-byte replaces: all three files mix CRLF and LF.

Run: .venv\\Scripts\\python.exe scratchpad\\quant_plan\\corruption\\_add_int2shift_candidate.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "scratchpad" / "quant_plan"
SWEEP = RUN / "quant_sweep.py"
GRID = RUN / "rust_grid.py"
CORRUPT = RUN / "corruption" / "check_corruption.py"

NAME = "l2i6s4n1i8s2"


def _nl(raw: bytes) -> bytes:
    return b"\r\n" if raw.count(b"\r\n") * 2 > raw.count(b"\n") else b"\n"


SWEEP_ANCHOR = (
    b'                     "(1,024/2,112 = 3,136 = 256 BELOW row 35, the smallest passing row)",\n'
    b'                     (1024, 2112)),\n'
)
SWEEP_NEW = (
    b'    "l2i6s4n1i8s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\n'
    b'                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int2"},\n'
    b'                     "card rank-2 int6 + int4 shifts and note rank-1 int8 + int2 shifts "\n'
    b'                     "(1,024/1,920 = 2,944 = 448 BELOW row 35) - gated second point of chain 21",\n'
    b'                     (1024, 1920)),\n'
)

GRID_ANCHOR = (
    b'    "l2i6s4n1i8s3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\n'
    b'                     "RWKV_QUANT_SHIFTS": "1",\n'
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int3"},\n'
)
GRID_NEW = (
    b'    # The next rung down: the note\'s int2 SHIFT bit for 192 fewer bits (2,944 total).\n'
    b'    "l2i6s4n1i8s2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int8",\n'
    b'                     "RWKV_QUANT_SHIFTS": "1",\n'
    b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\n'
)

COR_ANCHOR = (
    b'    "l2i6s4n1i8s3": (RUN / "qat_it36l2i6s4n1i8s3" / "qat_124.pth", "l2i6s4n1i8s3",\n'
    b'                     "chain 21 point 1 (int6 card + note int3 shifts, 3,136 bits)"),\n'
)
COR_NEW = COR_ANCHOR + (
    b'    "l2i6s4n1i8s2": (RUN / "qat_it36l2i6s4n1i8s2" / "qat_124.pth", "l2i6s4n1i8s2",\n'
    b'                     "chain 21 point 2 (int6 card + note int2 shifts, 2,944 bits)"),\n'
)


def patch(path: Path, old: bytes, new: bytes, label: str) -> int:
    raw = path.read_bytes()
    nl = _nl(raw)
    if nl != b"\n":
        old = old.replace(b"\r\n", b"\n").replace(b"\n", nl)
        new = new.replace(b"\r\n", b"\n").replace(b"\n", nl)
    if NAME.encode() in raw:
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
    changed += patch(GRID, GRID_ANCHOR, GRID_ANCHOR + GRID_NEW, "CONFIGS entry")
    changed += patch(CORRUPT, COR_ANCHOR, COR_NEW, "CONFIGS entry")
    print(f"done ({changed} files changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())