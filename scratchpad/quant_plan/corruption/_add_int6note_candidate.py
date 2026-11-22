"""Register `l2i6s4n1i6s3` (2,752 bits) in quant_sweep.CANDIDATES and check_corruption.CONFIGS.

The 7-user deploy screen (rust_grid `it36wide5`) prices it at +0.000343 ahead / +0.000435 imm against
`l2i8s4` - both INSIDE the +0.0005 ceiling, at 640 bits below row 35 (the smallest passing row) and
384 below chain 21's 3,136-point. It is the first sub-3,136 rung whose screen sits inside the ceiling,
which is why it is worth a 750-user QAT run.

The screen also closes two neighbours: the int5 card costs +0.000403/+0.000803 at the same note (so
int5 is WORSE than int6, not a free step), and the note's int2 SHIFT rung costs about +0.0012-0.0014
imm on top of the int3 rung at either factor width.
"""
from pathlib import Path

SWEEP = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\quant_sweep.py")
CHECK = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\corruption\check_corruption.py")


def _nl(raw: bytes) -> bytes:
    return b"\r\n" if raw.count(b"\r\n") * 2 > raw.count(b"\n") else b"\n"


def add(path: Path, anchor: bytes, block: bytes, label: str) -> None:
    raw = path.read_bytes()
    if b"l2i6s4n1i6s3" in raw:
        print(f"{path.name}: entry already present")
        return
    eol = _nl(raw)
    a = anchor.replace(b"\n", eol)
    b = block.replace(b"\n", eol)
    n = raw.count(a)
    if n != 1:
        raise SystemExit(f"ABORT [{label}]: anchor found {n} times, expected 1")
    path.write_bytes(raw.replace(a, a + b, 1))
    print(f"{path.name}: {label} added")


SWEEP_ANCHOR = b'''    "l2i6s4n1i8s2": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int8",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int2"},
                     "card rank-2 int6 + int4 shifts and note rank-1 int8 + int2 shifts "
                     "(1,024/1,920 = 2,944 = 448 BELOW row 35) - gated second point of chain 21",
                     (1024, 1920)),
'''
SWEEP_BLOCK = b'''    "l2i6s4n1i6s3": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",
                      "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int3"},
                     "card rank-2 int6 + int4 shifts and note rank-1 int6 + int3 shifts "
                     "(1,024/1,728 = 2,752 = 640 BELOW row 35, and the only sub-3,136 rung whose "
                     "7-user deploy screen sits inside the +0.0005 ceiling on BOTH heads: "
                     "+0.000343/+0.000435)",
                     (1024, 1728)),
'''

CHECK_ANCHOR = b'''    "l2i6s4n1i8s2": (RUN / "qat_it36l2i6s4n1i8s2" / "qat_124.pth", "l2i6s4n1i8s2",
                     "chain 21 point 2 (int6 card + note int2 shifts, 2,944 bits)"),
'''
CHECK_BLOCK = b'''    "l2i6s4n1i6s3": (RUN / "qat_it36l2i6s4n1i6s3" / "qat_124.pth", "l2i6s4n1i6s3",
                     "chain 22 point 1 (int6 card + note int6 factors + note int3 shifts, 2,752 bits)"),
'''

add(SWEEP, SWEEP_ANCHOR, SWEEP_BLOCK, "CANDIDATES entry")
add(CHECK, CHECK_ANCHOR, CHECK_BLOCK, "CONFIGS entry")

# verify both registries still import and carry the expected bit counts
import sys

sys.path.insert(0, str(SWEEP.parent))
sys.path.insert(0, str(CHECK.parent))
import quant_sweep as qs

env, desc, bits = qs.CANDIDATES["l2i6s4n1i6s3"]
print("quant_sweep:", env, "bits", bits, "cands", len(qs.CANDIDATES))
print("  bits_for", qs.bits_for(env) if hasattr(qs, "bits_for") else "n/a")

import importlib.util

spec = importlib.util.spec_from_file_location("cc", CHECK)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
print("check_corruption: configs", len(cc.CONFIGS), "entry", cc.CONFIGS["l2i6s4n1i6s3"][1:])