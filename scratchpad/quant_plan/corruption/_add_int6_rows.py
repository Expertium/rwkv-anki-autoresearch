"""Add the int6/int5-factor rungs of the row-17 ladder to `rust_grid.CANDIDATES` (raw bytes).

WHY: every rung below row 17 that has been tested so far used int4 factors (rows 19-26). The `int6` and
`int5` rungs have NEVER been tested and they are cheap: `int6` saves 256 card + 384 note bits, `int5`
saves 384 + 576 -- i.e. 640 / 960 bits below row 17's 1,280/2,304 (3,584 total), or 18% / 27% smaller.

The Rust engine has always spoken int6/int5 (`rust/rwkv-infer/src/model.rs:2104-2105`, `parse_level`), and
`rust_grid.QBITS` already carries them, so a Rust PTQ probe on the EXISTING row-17 QAT checkpoint needs no
training and no code change. That probe is what this script enables: it prices the rungs before any GPU
time is spent.

The Python QAT path is the side that cannot express them yet: `rwkv/architecture.py:151` `_QMAX` holds only
int8/int4/int3/int2/fp32, so `card:2:int6` raises KeyError there. That is a separate gap, NOT touched here;
`rust_grid.py`'s own docstring (line 6) already records it.

Raw-bytes write because the edit tool matches LF while this file is CRLF-dominant, which has already failed
twice in this repo, and the anchor itself is LF while the surrounding file is CRLF. The newline for the
inserted text is taken FROM THE ANCHOR so the result stays consistent with its neighbours.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR_TMPL = (
    '    "l2i8n3i8": {{"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:3:int8", "RWKV_QUANT_SHIFTS": "1",{nl}'
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_FAST": "1"}},{nl}'
)

BODY = [
    "",
    "    # THE int6/int5 FACTOR RUNGS (2026-11-22). Every rung below row 17 that has been tested so",
    "    # far used int4 factors (rows 19-26); int6 and int5 were NEVER tried, although the ENGINE has",
    "    # always supported them (parse_level, model.rs:2104) and only the Python QAT path lacks the",
    "    # _QMAX keys. The card keeps rank 2 and the note rank 1 (row 17's exact layout), so the ONLY",
    "    # change against l2i8s4 is factor precision: 6 bits instead of 8. bits_for:",
    "    #   l2i6s4  card 2x2x32x6 + 64x4 = 1,024   note 3x(2x1x32x6 + 64x4) = 1,920   total 2,944",
    "    #   l2i5s4  card 2x2x32x5 + 64x4 =   896   note 3x(2x1x32x5 + 64x4) = 1,728   total 2,624",
    "    # i.e. 640 and 960 bits below row 17. The old rank-1 PTQ grid (l1i4/l1i5/l1i6/l1i8,",
    "    # RESULTS.md) said precision above int4 does not pay AT RANK 1, because the residual there IS",
    "    # the rank truncation; but that was rank 1, and PTQ on a checkpoint adapted to int8. At rank 2",
    "    # on a QAT-trained checkpoint the question is open.",
    '    "l2i6s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int6,note:1:int6", "RWKV_QUANT_SHIFTS": "1",',
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"},',
    '    "l2i5s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int5", "RWKV_QUANT_SHIFTS": "1",',
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"},',
]


def main() -> int:
    raw = RG.read_bytes()
    if b'"l2i6s4"' in raw:
        print("already patched (l2i6s4 present); nothing to do")
    else:
        nl = "\r\n" if ANCHOR_TMPL.format(nl="\r\n").encode() in raw else "\n"
        anchor = ANCHOR_TMPL.format(nl=nl).encode("utf-8")
        n = raw.count(anchor)
        assert n == 1, f"anchor count {n}, expected 1 -- refusing to patch"
        new = nl.join(BODY).encode("utf-8") + nl.encode()
        RG.write_bytes(raw.replace(anchor, anchor + new))
        print(f"patched {RG.name} (+{len(new)} bytes, newline={nl!r})")

    import importlib.util
    spec = importlib.util.spec_from_file_location("rust_grid", RG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("l2i8s4", "l2i6s4", "l2i5s4", "l2i4s4"):
        env = mod.CANDIDATES.get(name)
        if env is None:
            print(f"  {name}: NOT IN CANDIDATES")
            continue
        b = mod.bits_for(env)
        print(f"  {name}: card={b.get('card')}b note={b.get('note')}b total={sum(b.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
