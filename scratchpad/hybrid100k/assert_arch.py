"""Phase-0 guard for an ARCHITECTURE arm: is the model this process builds the model we priced?

WHY THIS EXISTS. `RWKV_STRIP_CMIX` matches on the literal string "<stream>:<layer_id>" and a name
that matches NO layer is SILENTLY IGNORED (`rwkv_model.py:578`). Every hybrid arm changes the
per-stream DEPTHS, so the champion's list -- which names `user_id:2`, `preset_id:2`, `deck_id:2`,
`card_id:1` -- half-applies on an arm that is shallower, leaving channel mixers in place that the
param count was computed without. Nothing raises. The banner still prints. The run trains a
different model than the one in the design doc and the ratio gate is then computed against the
wrong denominator.

Same family as the QAT env that was parsed and then discarded: a lever that is CONFIGURED is not a
lever that is APPLIED, and only the CONSUMED state can tell you which happened. So this asserts on
the built model, not on the environment.

Two checks, both cheap and both necessary:

  1. EXACT PARAM COUNT. One number that is sensitive to the arch file, every strip entry, the head
     config and the input width at once. An arm is priced once, by mk_arch.py, and must reproduce
     that number here or the run is not the experiment.
  2. EVERY STRIP ENTRY MATCHED A REAL LAYER. The param count would usually catch a mismatch too,
     but not always -- and when it does, it reports "wrong size" for what is really "your strip
     list is stale". This names the offending entry instead.

Usage:  assert_arch.py <expected_params>
Exit 0 = the model is the priced one. Exit 46 = it is not. CPU-only, a few seconds, no GPU.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    if len(sys.argv) != 2:
        print("usage: assert_arch.py <expected_params>")
        return 2
    want = int(sys.argv[1])

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as CFG
    from rwkv.model.srs_model import SrsRWKV

    # --- check 2 first: it is the cheaper failure to explain ---------------------------------
    # Valid targets are (stream_name, layer_id) for every layer that actually exists. The deck
    # tree renames a repeated stream to `deck_id@k` but the env hooks see it as `deck_id`, so
    # strip the suffix before comparing (arm A has no tree; this keeps the guard general).
    valid = set()
    for name, mcfg in CFG.modules:
        base = name.split("@")[0]
        for j in range(mcfg.n_layers):
            valid.add(f"{base}:{j}")
    entries = [t.strip() for t in os.environ.get("RWKV_STRIP_CMIX", "").split(",") if t.strip()]
    unmatched = [e for e in entries if e not in valid]
    if unmatched:
        print(f"[assert-arch] FAIL: RWKV_STRIP_CMIX entries match no layer: {unmatched}")
        print(f"[assert-arch]   the arch has: {sorted(valid)}")
        print("[assert-arch]   these are SILENTLY IGNORED, so the model is not the priced one.")
        return 46
    print(f"[assert-arch] all {len(entries)} RWKV_STRIP_CMIX entries matched a real layer")

    # --- check 1: the number ------------------------------------------------------------------
    model = SrsRWKV(CFG)
    got = sum(p.numel() for p in model.parameters())
    depths = ", ".join(f"{n}:{m.n_layers}" for n, m in CFG.modules)
    print(f"[assert-arch] d_model={CFG.d_model}  depths=({depths})")
    print(f"[assert-arch] params: got {got}  expected {want}")
    if got != want:
        print(f"[assert-arch] FAIL: param count differs by {got - want}. The arch file, the strip "
              "list, the head config or the input width is not what this arm was priced with.")
        return 46
    print("[assert-arch] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
