"""Add the CALIBRATION configs for the int6/int5 probe to `rust_grid.CONFIGS` (raw bytes).

The 2-user Rust PTQ probe above says int6 (1,024/1,920) and int5 (896/1,728) are free against l2i8s4.
Before that is believed, the probe itself has to be calibrated: two layouts in it already have
750-user QAT numbers in RESULTS.md, so the probe's own delta can be compared with the known delta.

  row 23 `l2i4s4`    card rank2 int4 + note rank1 int8, s4 =   768/2,304   known imm tax +0.000266
  row 21 `l2i8s4n4`  card rank2 int8 + note rank1 int4, s4 = 1,280/1,536   known imm tax +0.000638
  row 17 `l2i8s4`    card rank2 int8 + note rank1 int8, s4 = 1,280/2,304   known imm tax +0.000327

so the probe should reproduce these DELTAS against its own l2i8s4 run:
  card int8 -> int4  (l2i4s4   - l2i8s4)          = +0.000266 - +0.000327 = -0.000061
  note int8 -> int4  (l2i8s4n4 - l2i8s4)          = +0.000638 - +0.000327 = +0.000311

If the probe reproduces those magnitudes, the probe is a usable predictor and the int6/int5 near-zero
result is real. If the probe understates int4 badly, the int6/int5 result cannot be trusted either and
the rungs have to be settled by a QAT run on the 750-user roster.

Raw-bytes write (CRLF dominant there, LF in places) because the edit tool matches LF only.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR_TMPL = (
    '    "l2i5s4": {{"RWKV_STATE_LOWRANK_SCOPE": "card:2:int5,note:1:int5", "RWKV_QUANT_SHIFTS": "1",{nl}'
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"}},{nl}'
)

BODY = [
    "    # PROBE CALIBRATION (2026-11-22): l2i4s4 is row 23's layout and l2i8s4n4 is row 21's, and both",
    "    # have 750-user QAT numbers in RESULTS.md, so running all three in one grid makes the 2-user",
    "    # Rust PTQ probe checkable against a known delta.",
    '    "l2i4s4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int4,note:1:int8", "RWKV_QUANT_SHIFTS": "1",',
    '               "RWKV_STATE_SHIFT_LEVEL": "int4"},',
    '    "l2i8s4n4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int4", "RWKV_QUANT_SHIFTS": "1",',
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4"},',
]

WANT = ["plain", "l2i8s4", "l2i6s4", "l2i5s4", "l2i4s4", "l2i8s4n4"]


def main() -> int:
    raw = RG.read_bytes()
    if b'"l2i4s4"' in raw:
        print("calibration already present; nothing to do")
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
    missing = [n for n in WANT if n not in mod.CONFIGS]
    print("missing:", missing)
    for name in WANT:
        if name in mod.CONFIGS:
            b = mod.bits_for(mod.CONFIGS[name])
            print(f"  {name}: card={b.get('card')}b note={b.get('note')}b total={sum(b.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
