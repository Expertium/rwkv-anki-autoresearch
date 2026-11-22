"""Add `l2i4s4n4ef2`: both streams int4 + error feedback whose BUFFER is int2 (raw bytes).

WHY. The 15-user screen says EF itself is real (imm -0.002453, 95% [-0.004826, -0.000539], better on
8/15 users) but `l2i4s4n4ef` - EF at int4 factors with an int4 error buffer - is only LEVEL with row 17
(ahead -0.000134 and imm -0.000819, both intervals spanning zero) for 256 fewer bits. The buffer is
what makes EF expensive: it is real state, 256 b/layer at int4, so the card pays 256 and the three note
layers pay 768.

The engine's own note says the carried error's bias is low-rank ("rank-1 may suffice") and offers
`RWKV_EF_PQ` to get it to ~96 b. So the natural next question is whether the buffer survives at a
COARSER width, which is where the bits are:

    card  rank-2 int4  512 + shifts 256 + e int2 128  =   896
    note  x3 rank-1 int4 768 + shifts 768 + e int2 384 = 1,920
                                                total   = 2,816 bits

That is 768 BELOW row 17 (-21%) and exactly row 21's bit count - and row 21 (`l2i8s4n4`, note at int4
with no EF) failed the gate by 0.00014. So this point asks directly whether error feedback can rescue
the layout that just missed: if it can, the axis-1 best moves from 3,584 to 2,816 bits in one step,
which is more than the whole chain-17 int6 cut (640).

Reported bits must count the buffer as state, so `bits_for` already handles it; this only adds the
config. PTQ on the int8-adapted checkpoint, i.e. a screen, not a gate result - and note the different
width for U/V is NOT involved here (no VLEVEL), the state and buffer are symmetric int4 / int2.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR_TMPL = (
    '    "l2i4s4n4ef": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",\n'
    '                   "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_EF": "1",\n'
    '                   "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int4"},\n'
)

BODY = [
    "    # The same layout with the ERROR BUFFER at int2 instead of int4: 384 fewer state bits, so",
    "    # 2,816 total - 768 below row 17 and exactly row 21's size, the layout that failed by 0.00014.",
    '    "l2i4s4n4ef2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int4", "RWKV_QUANT_SHIFTS": "1",',
    '                    "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_EF": "1",',
    '                    "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int2"},',
]


def main() -> int:
    raw = RG.read_bytes()
    if b'"l2i4s4n4ef2"' in raw:
        print("already present; nothing to do")
    else:
        for nl in ("\r\n", "\n"):
            ob = ANCHOR_TMPL.replace("\n", nl).encode("utf-8")
            if raw.count(ob) == 1:
                ins = ob + nl.join(BODY).encode("utf-8") + nl.encode("utf-8")
                RG.write_bytes(raw.replace(ob, ins))
                print(f"patched (newline={nl!r})")
                break
        else:
            raise SystemExit("ABORT: no single-newline match for the l2i4s4n4ef anchor")

    import importlib.util
    spec = importlib.util.spec_from_file_location("rust_grid", RG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("l2i8s4", "l2i4s4n4ef", "l2i4s4n4ef2"):
        b = mod.bits_for(mod.CONFIGS[name])
        print(f"  {name}: card={b['card']}b note={b['note']}b total={sum(b.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())