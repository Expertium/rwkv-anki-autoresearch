"""Report parameter count and per-card RNN state size for an AnkiRWKVConfig.

State-size metric (protocol point 5): the per-card state Anki must persist = the
card_id stream's RNN state (the other streams' states are shared per note/deck/preset/
user). Per layer that's the time-mix WKV matrix (H*K*K) + time/channel token-shifts
(2*d_model). This must not increase across optimization iterations.

Usage: python optimization/model_stats.py            # default architecture
       (import build_report(config) for a custom AnkiRWKVConfig)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model_rnn import SrsRWKVRnn

STREAM_NAMES = ["card", "deck", "note", "preset", "user"]


def stream_state_floats(cfg):
    """Floats of RNN state for one layer = H*K*K (WKV) + 2*d_model (token shifts)."""
    H = cfg.n_heads
    K = cfg.d_model // cfg.n_heads
    per_layer = H * K * K + 2 * cfg.d_model
    return per_layer, cfg.n_layers * per_layer


def build_report(anki_cfg=DEFAULT_ANKI_RWKV_CONFIG):
    model = SrsRWKVRnn(anki_cfg)
    total_params = sum(p.numel() for p in model.parameters())

    # state per stream
    state = {}
    for (name, cfg), sname in zip(anki_cfg.modules, STREAM_NAMES):
        per_layer, total = stream_state_floats(cfg)
        state[sname] = total
    card_state = state["card"]  # the per-card persisted state
    all_state = sum(state.values())

    report = {
        "total_params": total_params,
        "card_state_floats": card_state,
        "card_state_bytes_f32": card_state * 4,
        "card_state_kib_f32": round(card_state * 4 / 1024, 3),
        "all_streams_state_floats": all_state,
        "per_stream_state_floats": state,
        "d_model": anki_cfg.d_model,
        "n_heads": anki_cfg.modules[0][1].n_heads,
        "layers": [c.n_layers for _, c in anki_cfg.modules],
        "channel_mixer_factors": [c.channel_mixer_factor for _, c in anki_cfg.modules],
    }
    return report


def main():
    r = build_report()
    print(f"total params        : {r['total_params']:,}")
    print(f"d_model / n_heads    : {r['d_model']} / {r['n_heads']}  (K={r['d_model']//r['n_heads']})")
    print(f"layers per stream    : {r['layers']}  (card,deck,note,preset,user)")
    print(f"channel_mixer_factors: {r['channel_mixer_factors']}")
    print(f"per-stream state floats: {r['per_stream_state_floats']}")
    print(f"PER-CARD state       : {r['card_state_floats']:,} floats = "
          f"{r['card_state_bytes_f32']:,} B = {r['card_state_kib_f32']} KiB (f32)")
    print(f"all-streams state    : {r['all_streams_state_floats']:,} floats")


if __name__ == "__main__":
    main()
