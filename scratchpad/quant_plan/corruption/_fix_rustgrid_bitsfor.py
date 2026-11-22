"""One-shot: rewrite rust_grid.bits_for so it agrees with quant_sweep.bits_for.

`rust_grid.py` has mixed line endings (180 CRLF / 13 LF), so the edit tool cannot match a multi-line
block in it. This helper locates `def bits_for` by index and replaces the whole function body up to
the next top-level `def`.

The bug being fixed: a BARE `RWKV_STATE_SHIFT_LEVEL` (e.g. "int4", which is what every deploy config
uses, because the engine's `parse_level` takes one global level) was read as *note-only* by the old
`key = "card" if "card" in entry else "note"` line. The card then fell back to its WKV width, so
`l2i8s4` was reported as 1,536 card bits instead of the true 1,280.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"

NEW = '''def bits_for(env):
    """State bits per stream (WKV + token shifts). Must agree with `quant_sweep.bits_for`.

    A BARE `RWKV_STATE_SHIFT_LEVEL` is GLOBAL: the engine's `parse_level` takes ONE level and applies
    it to every stream. Reading a bare level as "note-only" (the old behaviour here) made the card
    fall back to its WKV width and reported 1,536 bits for `l2i8s4` instead of the true 1,280, so the
    deployed row was scored against the wrong card bit count. A `stream:level` list is accepted too,
    for symmetry with the training side, but the engine ignores such a string.
    """
    shift = {}
    bare = None
    for entry in env.get("RWKV_STATE_SHIFT_LEVEL", "").split(","):
        entry = entry.strip()
        if not entry:
            continue  # "" splits to [""] and would otherwise register a bogus fp32 shift level
        head, level = entry.split(":")[0], entry.split(":")[-1]
        if ":" in entry:
            shift[head] = QBITS.get(level, 32)
        else:
            bare = QBITS.get(level, 32)
    low = {}
    for entry in env.get("RWKV_STATE_LOWRANK_SCOPE", "").split(","):
        parts = entry.split(":")
        if len(parts) >= 3:
            low[parts[0]] = (int(parts[1]), QBITS[parts[2]])
    full = {}
    for entry in env.get("RWKV_STATE_QUANT_SCOPE", "").split(","):
        parts = entry.split(":")
        if parts[0] in LAYERS:
            full[parts[0]] = QBITS.get(parts[-1], 32)
    out = {}
    for s, n in LAYERS.items():
        # the shifts follow the WKV level when no shift override is given
        if bare is not None:
            sb = bare
        else:
            sb = shift.get(s, low[s][1] if s in low else full.get(s, 32))
        if s in low:
            r, qb = low[s]
            out[s] = n * (2 * r * 32 * qb + 64 * sb)
        else:
            out[s] = n * (1024 * full.get(s, 32) + 64 * sb)
    return out


'''

text = P.open(encoding="utf-8", newline="").read()
start = text.index("def bits_for(env):")
m = re.compile(r"\r?\ndef ").search(text, start + 1)
if not m:
    raise SystemExit("could not find the end of bits_for")
end = m.start() + 1
old = text[start:end]
nl = "\r\n" if old.count("\r\n") > old.count("\n") - old.count("\r\n") else "\n"
new = NEW.replace("\n", nl)
text = text[:start] + new.rstrip("\r\n") + nl + text[end:]
P.open("w", encoding="utf-8", newline="").write(text)
print(f"replaced {len(old)} chars with {len(new)} chars (newline {nl!r})")

sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402
import quant_sweep as qs  # noqa: E402

print("\nrust_grid.bits_for on every CONFIGS entry:")
for name, env in rg.CONFIGS.items():
    print(f"  {name:12s} {rg.bits_for(env)}")
print("\nthe four rows that matter (must match quant_sweep.bits_for on the training side):")
for rname, cname in (("l2i8", "l2i8"), ("l2i8s4", "l2i8s4"), ("l3i8s4", "l3i8s4"),
                     ("l2i8n2i8", "l2i8n2i8"), ("l3i8n2i8", "l3i8n2i8"), ("l2i8n3i8", "l2i8n3i8")):
    if cname not in qs.CANDIDATES:
        print(f"  {rname:12s} rust={rg.bits_for(rg.CONFIGS[rname])}  (no quant_sweep candidate)")
        continue
    env, _c, bits_override = qs.CANDIDATES[cname]
    train = {"card": bits_override[0], "note": bits_override[1]} if bits_override else qs.bits_for(env)
    rust = rg.bits_for(rg.CONFIGS[rname])
    ok = train == rust
    print(f"  {rname:12s} rust={rust} train={train} agree={ok}")
