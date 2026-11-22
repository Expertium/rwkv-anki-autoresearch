"""One-shot: correct ONLY the `bits` fields that the fixed `rust_grid.bits_for` bug explains.

`rust_grid.bits_for` read a BARE `RWKV_STATE_SHIFT_LEVEL` as note-only, so the card fell back to its
WKV width: `l2i8s4` was written as 1,536 card bits instead of 1,280. The function is fixed.

An entry is rewritten ONLY when BOTH hold:
  * the recorded `bits` equals what the OLD (buggy) function computes for today's env, AND
  * it differs from what the FIXED function computes.
Anything else is drift with another cause (an older config definition, e.g. the note field of
`l1i4` in `rust_grid_small.json` reads 6,912 = rank-1 int4 with 32-bit shifts, which the bug cannot
produce) and it is LEFT ALONE and listed, because overwriting it would falsify a recorded number.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402


def buggy_bits_for(env):
    """The pre-fix version, kept only to identify which records the fix explains."""
    shift = {}
    for entry in env.get("RWKV_STATE_SHIFT_LEVEL", "").split(","):
        if not entry.strip():
            continue
        key = "card" if "card" in entry else "note"
        level = entry.split(":")[-1] if ":" in entry else entry
        shift[key] = rg.QBITS.get(level, 32)
    low = {}
    for entry in env.get("RWKV_STATE_LOWRANK_SCOPE", "").split(","):
        parts = entry.split(":")
        if len(parts) >= 3:
            low[parts[0]] = (int(parts[1]), rg.QBITS[parts[2]])
    full = {}
    for entry in env.get("RWKV_STATE_QUANT_SCOPE", "").split(","):
        parts = entry.split(":")
        if parts[0] in rg.LAYERS:
            full[parts[0]] = rg.QBITS.get(parts[-1], 32)
    out = {}
    for s, n in rg.LAYERS.items():
        sb = shift.get(s, low[s][1] if s in low else full.get(s, 32))
        if s in low:
            r, qb = low[s]
            out[s] = n * (2 * r * 32 * qb + 64 * sb)
        else:
            out[s] = n * (1024 * full.get(s, 32) + 64 * sb)
    return out


fixed, unexplained, unknown = [], [], []
for p in sorted((ROOT / "scratchpad" / "quant_plan").glob("rust_grid_*.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    touched = {}
    for name, rec in data.items():
        if name not in rg.CONFIGS or not isinstance(rec, dict) or "bits" not in rec:
            unknown.append(f"{p.name}:{name}")
            continue
        env = rg.CONFIGS[name]
        new = rg.bits_for(env)
        if rec["bits"] == new:
            continue
        if rec["bits"] == buggy_bits_for(env):
            touched[name] = (rec["bits"], new)
            rec["bits"] = new
        else:
            unexplained.append(f"{p.name}:{name} recorded={rec['bits']} fixed={new} "
                               f"buggy={buggy_bits_for(env)}")
    if touched:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        for name, (old, new) in touched.items():
            fixed.append(f"{p.name}:{name} {old} -> {new}")

print("corrected (the fixed bug explains the recorded value):")
for line in fixed or ["  (none)"]:
    print("  " + line)
print("left alone (drift with another cause):")
for line in unexplained or ["  (none)"]:
    print("  " + line)
print("entries whose config name is not in rust_grid.CONFIGS:")
for line in unknown or ["  (none)"]:
    print("  " + line)
