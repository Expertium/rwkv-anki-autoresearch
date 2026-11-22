"""One-shot: record the E2 fast rank-r solver in `rust_grid.CONFIGS`.

`RWKV_LOWRANK_FAST=1` is a SOLVER choice, not a bit-budget change (`bits_for` ignores it), and it is
what makes a rank>=2 NOTE deployable (rev/s 0.618 -> 0.803 measured). It belongs in the config dict so
every tool that describes the deploy config -- the finiteness check, the speed probe, the scorer --
describes what actually ships. Parity runs must NOT set it: the fast factors are an approximation.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
p = ROOT / "scratchpad" / "quant_plan" / "rust_grid.py"
text = p.open(encoding="utf-8", newline="").read()

OLD_COMMENT = """    # Rank-2 (and 3) on the NOTE, the new chain-14 candidates (Andrew, 2026-11-22 12:01). The engine's
    # low-rank path is generic over the stream, so these exist to prove it parses a rank>=2 note and to
    # price the speed. l2i8n2i8 is the direct twin of l2i8s4 with the note's rank raised 1 -> 2."""

NEW_COMMENT = """    # Rank-2 (and 3) on the NOTE, the new chain-14 candidates (Andrew, 2026-11-22 12:01). The engine's
    # low-rank path is generic over the stream, so these exist to prove it parses a rank>=2 note and to
    # price the speed. l2i8n2i8 is the direct twin of l2i8s4 with the note's rank raised 1 -> 2.
    # RWKV_LOWRANK_FAST=1 is the FAST rank-r SOLVER (E2, 2026-11-22): a deploy-side approximation, not a
    # bit-budget change (bits_for ignores it). Without it a rank>=2 NOTE measures 0.618x rev/s (FAIL on
    # the 0.75x floor); with it, 0.803x for l2i8n2i8 and 0.791x for l3i8n2i8, at a deploy log-loss cost
    # of imm +0.000051 / ahead +0.000000 on users 107/136/156 (rust_fastrank_check.json,
    # rust_score.json, rust_speed_fastrank_{on,off}.json). Do NOT set it for a Python<->Rust PARITY run:
    # the fast factors are not bit-identical to the reference truncation by construction."""

if OLD_COMMENT not in text:
    print("ABORT: the config comment is not where it was")
    sys.exit(1)
text = text.replace(OLD_COMMENT, NEW_COMMENT)
print("ok [comment]")

SCOPES = ["card:2:int8,note:2:int8", "card:3:int8,note:2:int8", "card:2:int8,note:3:int8"]
for scope in SCOPES:
    pat = re.compile(r'("RWKV_STATE_LOWRANK_SCOPE": "' + re.escape(scope) + r'",.*?)\}', re.S)
    m = pat.search(text)
    if not m:
        print(f"ABORT: no config entry for {scope}")
        sys.exit(1)
    text = text[:m.end(1)] + ', "RWKV_LOWRANK_FAST": "1"}' + text[m.end():]
    print(f"ok [fast flag] {scope}")

p.open(encoding="utf-8", newline="", mode="w").write(text)

# verify by importing
import importlib.util
spec = importlib.util.spec_from_file_location("rg_check", p)
rg = importlib.util.module_from_spec(spec)
sys.modules["rg_check"] = rg
spec.loader.exec_module(rg)
for name in ("l2i8n2i8", "l3i8n2i8", "l2i8n3i8", "l2i8s4", "l2i8"):
    env = rg.CONFIGS[name]
    print(f"  {name:10s} fast={env.get('RWKV_LOWRANK_FAST', '-')} bits={rg.bits_for(env)}")
