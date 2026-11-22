"""Add the FIXED-BASIS (codec) deploy configs + bit accounting to rust_grid.py, and refresh the two
comments that claimed the engine cannot do a per-stream shift level (it can now).

bits_for for a fixed-basis stream: n_layers * (m*m*qbits + 64*shift_bits), the same formula
quant_sweep.bits_for uses for the fb candidates -- the payload is an m x m coefficient block, not
2*K*r factors.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

BASIS = "scratchpad/quant_plan/corruption/fixedbasis_l2i8s4all.txt"

CONFIGS_ANCHOR = '''                    "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int2"},

}'''

CONFIGS_NEW = '''                    "RWKV_EF_ERANK": "1", "RWKV_EF_ELEVEL": "int2"},

    # FIXED-BASIS (codec) state coding (approach 1, 2026-11-22): store the m x m coefficient block
    # C = P^T W Q against a FIXED K x m basis that ships with the model (model.rs::fixedbasis_roundtrip),
    # instead of the rank-r path's 2*K*r factor numbers. Basis file = the fitted per-(stream, layer)
    # bases (corruption/fixedbasis_l2i8s4all.txt, fitted on 3 users of the row-17 checkpoint).
    #   fb8i8      m=8 int8 on BOTH streams        768 + 3x768 = 3,072
    #   l2i8fb8n   card rank-2 int8 + note fb m=8  1,280 + 3x768 = 3,584  (row 17's exact size)
    #   l2i8fb6n   card rank-2 int8 + note fb m=6  1,280 + 3x544 = 2,912  (672 below row 17)
    # l2i8fb8n is the clean codec A/B at equal bits; l2i8fb6n is the size claim.
    "fb8i8": {"RWKV_STATE_FIXEDBASIS_SCOPE": "card:8:int8,note:8:int8", "RWKV_QUANT_SHIFTS": "1",
              "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int4", "RWKV_FIXEDBASIS_PATH": BASIS_PATH},
    "fb4i8": {"RWKV_STATE_FIXEDBASIS_SCOPE": "card:4:int8,note:4:int8", "RWKV_QUANT_SHIFTS": "1",
              "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int4", "RWKV_FIXEDBASIS_PATH": BASIS_PATH},
    "l2i8fb8n": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_FIXEDBASIS_SCOPE": "note:8:int8", "RWKV_FIXEDBASIS_PATH": BASIS_PATH,
                 "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int4"},
    "l2i8fb6n": {"RWKV_STATE_LOWRANK_SCOPE": "card:2:int8", "RWKV_QUANT_SHIFTS": "1",
                 "RWKV_STATE_FIXEDBASIS_SCOPE": "note:6:int8", "RWKV_FIXEDBASIS_PATH": BASIS_PATH,
                 "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int4"},

}'''

POP_OLD = '''              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT"):'''
POP_NEW = '''              "RWKV_LOWRANK_COMPAND", "RWKV_LOWRANK_VLEVEL", "RWKV_STATE_QUANT",
              "RWKV_STATE_FIXEDBASIS_SCOPE", "RWKV_FIXEDBASIS_PATH"):'''

FB_PARSE_OLD = '''    ef_bits = 2 * _ef_rank * 32 * _ef_bits
    out = {}'''
FB_PARSE_NEW = '''    ef_bits = 2 * _ef_rank * 32 * _ef_bits
    # RWKV_STATE_FIXEDBASIS_SCOPE="note:8:int8": stream -> (m, coefficient qbits). The payload is an
    # m x m coefficient block instead of 2*K*r factors, so the WKV bits are m*m*qbits.
    fb = {}
    for entry in env.get("RWKV_STATE_FIXEDBASIS_SCOPE", "").split(","):
        parts = entry.strip().split(":")
        if len(parts) >= 2 and parts[0]:
            fb[parts[0]] = (int(parts[1]), QBITS.get(parts[-1], 32))
    out = {}'''

FB_LOOP_OLD = '''        if s in low:
            r, qb = low[s]'''
FB_LOOP_NEW = '''        if s in fb:
            m_, qb_ = fb[s]
            out[s] = n * (m_ * m_ * qb_ + 64 * sb)
        elif s in low:
            r, qb = low[s]'''

DOC_OLD = '''    A BARE `RWKV_STATE_SHIFT_LEVEL` is GLOBAL: the engine's `parse_level` takes ONE level and applies
    it to every stream. Reading a bare level as "note-only" (the old behaviour here) made the card
    fall back to its WKV width and reported 1,536 bits for `l2i8s4` instead of the true 1,280, so the
    deployed row was scored against the wrong card bit count. A `stream:level` list is accepted too,
    for symmetry with the training side, but the engine ignores such a string.'''
DOC_NEW = '''    A BARE `RWKV_STATE_SHIFT_LEVEL` is GLOBAL (every compressed stream); a `stream:level` list sets
    only the named streams (engine: model.rs, `shift_level`, 2026-11-22 - it used to be a single level
    and a mixed string was silently ignored, which is why rows 31/34/37/35 were loss-only). Reading a
    bare level as "note-only" (the old behaviour here) made the card fall back to its WKV width and
    reported 1,536 bits for `l2i8s4` instead of the true 1,280, so the deployed row was scored against
    the wrong card bit count.'''

ENGINE_NOTE_OLD = '''    # NOTE: the engine has NO per-stream shift level - parse_level() takes a bare "int4"/"int8" and any
    # "card:x,note:y" string yields None (= no override). So `l2i8n2i8s8` (card int4 / note int8 shifts)
    # cannot be expressed on the deploy side yet; that is a small engine gap, recorded in GPU_RUN_BOARD.'''
ENGINE_NOTE_NEW = '''    # NOTE: a PER-STREAM shift level now works (model.rs `shift_level`, 2026-11-22), so `l2i8n2i8s8`
    # (card int4 / note int8 shifts) can be expressed here as "card:int4,note:int8". The config below
    # still says "int4" for both, so it is NOT the s8 twin; add a separate entry if that matters.'''

raw = GRID.read_text(encoding="utf-8")
for label, old, new in (("CONFIGS", CONFIGS_ANCHOR, CONFIGS_NEW), ("pop", POP_OLD, POP_NEW),
                        ("fb parse", FB_PARSE_OLD, FB_PARSE_NEW), ("fb loop", FB_LOOP_OLD, FB_LOOP_NEW),
                        ("doc", DOC_OLD, DOC_NEW), ("engine note", ENGINE_NOTE_OLD, ENGINE_NOTE_NEW)):
    n = raw.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor matched {n} times (need 1)")
    raw = raw.replace(old, new, 1)
raw = raw.replace("BASIS_PATH", f'"{BASIS}"')
GRID.write_text(raw, encoding="utf-8")
print(f"patched {GRID} ({len(raw)} bytes)")
