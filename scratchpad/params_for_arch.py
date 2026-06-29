"""Compute total params + per-card/per-note state (fp32 AND deployed int8/int4/int2) for an arch
snapshot file WITHOUT touching the live rwkv/architecture.py.

Usage: python scratchpad/params_for_arch.py optimization/arch_snapshots/arch_iter42.py

NOTE on the two state numbers:
  - fp32 = the un-quantized arch property (what model_stats.py reports). It does NOT change with
    the deploy-time quant; it's just floats x 4 bytes.
  - int8/int4/int2 = the DEPLOYED state size once the WKV state is quantized at inference. The
    iter44 champion deploys card int2 + NOTE int2, so the card state Anki actually persists is the
    int2 figure (0.27 KiB) and note is 0.80 KiB, NOT the 4.25/12.75 KiB fp32 figures.
Convention (matches the recorded figures): deployed KiB = floats * bits / 8 / 1024.
"""
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimization.model_stats import build_report  # noqa: E402

BITS = {"fp32": 32, "int8": 8, "int4": 4, "int2": 2}


def deploy_kib(floats, level):
    return floats * BITS[level] / 8 / 1024


def fmt(floats):
    return "  ".join(f"{lvl} {deploy_kib(floats, lvl):.2f} KiB" for lvl in BITS)


for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("arch_snap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.DEFAULT_ANKI_RWKV_CONFIG
    r = build_report(cfg)
    per_stream = r["per_stream_state_floats"]
    card_f = per_stream["card"]
    note_f = per_stream["note"]
    print(f"{path}")
    print(f"  layers (card,deck,note,preset,user) = {r['layers']}")
    print(f"  total params  = {r['total_params']:,}")
    print(f"  per-CARD state ({card_f:,} floats): {fmt(card_f)}")
    print(f"  per-NOTE state ({note_f:,} floats): {fmt(note_f)}")
    print(f"  -> champion deploy (iter44: card int2 + note int2): "
          f"card {deploy_kib(card_f, 'int2'):.2f} KiB + note {deploy_kib(note_f, 'int2'):.2f} KiB "
          f"= {deploy_kib(card_f, 'int2') + deploy_kib(note_f, 'int2'):.2f} KiB/entity-pair")
