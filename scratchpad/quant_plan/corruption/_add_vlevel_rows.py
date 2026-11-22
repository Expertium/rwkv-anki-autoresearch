"""Add the ASYMMETRIC FACTOR BUDGET probe (RWKV_LOWRANK_VLEVEL) to `rust_grid` (raw bytes).

WHY. The engine splits the low-rank state into two quantized factors and, by default, uses ONE
bit-width for both. `RWKV_LOWRANK_VLEVEL` overrides the SECOND factor (V) independently of the scope
level, so a rank-r state can spend MORE bits on U and FEWER on V:

    rank r, U int8 / V int4:  r*32*8 + r*32*4 = 384r bits   instead of 512r

At row 17's exact layout that is a real axis-1 cut and it costs no new bits for scales:

    card rank 2, U int8 V int4:  2*32*8 + 2*32*4 =   768  + shifts 256 = 1,024   (vs 1,280)
    note rank 1, U int8 V int4:  3*(32*8 + 32*4 + 256)             = 1,920         (vs 2,304)
                                                                    2,944 total, 640 below row 17

V=int3 gives 2,784 and V=int2 gives 2,624. It STACKS with the chain-17 factor rungs, e.g. U int6
V int4 = 2,624 bits.

This has never been tried, even though the engine has implemented it (see the `lowrank_vqmax` field
and the `RWKV_LOWRANK_VLEVEL` env read in `rust/rwkv-infer/src/model.rs`). The old grid only ever
compared SYMMETRIC levels, where the finding was that precision above int4 does not pay at rank 1
(`l1i4`/`l1i5`/`l1i6`/`l1i8` all sat together); asymmetry is a different question, because U and V
are not interchangeable - U holds the state-space directions, V the read-out weights.

Two edits, both additive:
  1. `bits_for` learns VLEVEL, otherwise the reported bit counts would be the symmetric ones;
  2. three configs at row 17's layout for V int4/int3/int2.

The Python QAT path has NO equivalent, so these are PTQ measurements on the int8-adapted checkpoint -
i.e. a SCREEN, not a gate result. If V int4 turns out cheap, teaching `_QMAX`-style support to
`fake_lowrank_state` (one more per-stream value, the same shape of change as int6/int5) is what would
make it a trainable rung.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RG = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

ANCHOR_TMPL = (
    '    "l2i8s4n4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int4", "RWKV_QUANT_SHIFTS": "1",\n'
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4"},\n'
)

BODY = [
    "    # ASYMMETRIC FACTOR BUDGETS (2026-11-22): U keeps the scope level, V drops to VLEVEL. At",
    "    # row 17's layout U int8 / V int4 is 2,944 bits and V int3 / V int2 give 2,784 / 2,624,",
    "    # i.e. 640 / 800 / 960 bits below row 17. U and V are not interchangeable (U holds the",
    "    # state-space directions, V the read-out weights), which is why this is worth asking even",
    "    # though the symmetric ladder said precision above int4 does not pay at rank 1.",
    '    "l2i8s4v4": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",',
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int4"},',
    '    "l2i8s4v3": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",',
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int3"},',
    '    "l2i8s4v2": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8,note:1:int8", "RWKV_QUANT_SHIFTS": "1",',
    '                 "RWKV_STATE_SHIFT_LEVEL": "int4", "RWKV_LOWRANK_VLEVEL": "int2"},',
]

BITS_OLD = (
    '    out = {}\n'
    '    for s, n in LAYERS.items():\n'
    '        # the shifts follow the WKV level when no shift override is given\n'
)
BITS_NEW = (
    '    # RWKV_LOWRANK_VLEVEL overrides the SECOND factor (V) only, so the pair costs r*32*(U+V)\n'
    '    # bits instead of 2*r*32*U. Without this the reported bit counts would be the symmetric\n'
    '    # ones, which is exactly the kind of mis-count that made `l2i8s4` read 1,536 instead of 1,280.\n'
    '    vbits = QBITS.get(str(env.get("RWKV_LOWRANK_VLEVEL", "")).strip())\n'
    '    out = {}\n'
    '    for s, n in LAYERS.items():\n'
    '        # the shifts follow the WKV level when no shift override is given\n'
)

COMPUTE_OLD = (
    '        if s in low:\n'
    '            r, qb = low[s]\n'
    '            out[s] = n * (2 * r * 32 * qb + 64 * sb)\n'
)
COMPUTE_NEW = (
    '        if s in low:\n'
    '            r, qb = low[s]\n'
    '            if vbits is None:\n'
    '                out[s] = n * (2 * r * 32 * qb + 64 * sb)\n'
    '            else:\n'
    '                out[s] = n * (r * 32 * qb + r * 32 * vbits + 64 * sb)\n'
)


def main() -> int:
    raw = RG.read_bytes()
    if b'"l2i8s4v4"' in raw:
        print("VLEVEL configs already present; nothing to do")
    else:
        def patch_one(label, tmpl, new_text):
            """Each anchor carries ITS OWN newline: the config block is LF, `bits_for` is CRLF."""
            for nl in ("\r\n", "\n"):
                ob = tmpl.replace("\n", nl).encode("utf-8")
                if raw.count(ob) != 1:      # noqa: B023
                    continue
                if new_text is None:
                    ins = ob + nl.join(BODY).encode("utf-8") + nl.encode("utf-8")
                else:
                    ins = new_text.replace("\n", nl).encode("utf-8")
                print(f"  {label}: patched (newline={nl!r})")
                return raw.replace(ob, ins)  # noqa: B023
            raise SystemExit(f"ABORT [{label}]: no single-newline match found")

        raw = patch_one("configs", ANCHOR_TMPL, None)
        raw = patch_one("bits_for decl", BITS_OLD, BITS_NEW)
        raw = patch_one("bits_for compute", COMPUTE_OLD, COMPUTE_NEW)
        RG.write_bytes(raw)

    import importlib.util
    spec = importlib.util.spec_from_file_location("rust_grid", RG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("l2i8s4", "l2i8s4v4", "l2i8s4v3", "l2i8s4v2"):
        if name not in mod.CONFIGS:
            print(f"  {name}: MISSING")
            continue
        b = mod.bits_for(mod.CONFIGS[name])
        print(f"  {name}: card={b['card']}b note={b['note']}b total={sum(b.values())}")

    # bits_for must still give the known-correct numbers for the existing configs
    for name, want in (("l2i8s4", (1280, 2304)), ("l2i6s4", (1024, 1920)), ("l2i5s4", (896, 1728)),
                       ("l2i4s4n4", (768, 1536)), ("l2i4s4", (768, 2304))):
        b = mod.bits_for(mod.CONFIGS[name])
        got = (b["card"], b["note"])
        print(f"  regress {name}: {got} {'OK' if got == want else 'MISMATCH expected ' + str(want)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
