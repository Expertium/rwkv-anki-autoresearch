# TRACK-2 A15 (2026-07-23), the WIDTH cut: d_model 128 -> 96 via N_HEADS 4 -> 3
# (3 heads x K=32 -- keeps the per-head kernel shape; the WKV kernel is K-dynamic).
# Depth/strips unchanged (card 2L, deck 4L, note 1L, preset 3L, user 3L -- all
# floors; 9 mixer strips). LoRA dims stay at the A14 halving (decay/a/gate 8,
# v0-mix 4) -- adjust to 16/16/8/16 if A14 rejects. Delegated by Andrew 2026-07-23
# ("I'll leave it up to you... eventually we'll have to do it" -- the >=5x path).
# Everything downstream (input FC, GRU head, srs_heads) scales with d_model.
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 3  # A15: d_model = 32*3 = 96 (was 4 -> 128)

DROPOUT = 0.02
DROPOUT_LONG = 0.05
DROPOUT_LAYER = 0.01


@dataclass
class AnkiRWKVConfig:
    d_model: int
    modules: list
    dropout: float
    num_curves: int = 128
    num_points: int = 128
    head_fc_mult: int = 4
    features_fc_mult: int = 4


def _m(n_layers, cmf, dropout):
    return RWKV7Config(
        d_model=32 * N_HEADS, n_heads=N_HEADS, n_layers=n_layers,
        layer_offset=0, total_layers=n_layers, channel_mixer_factor=cmf,
        decay_lora=8, a_lora=8, v0_mix_amt_lora=4, gate_lora=8,  # A14: all halved
        dropout=dropout, dropout_layer=DROPOUT_LAYER,
    )


_layers = [
    ("card_id", _m(2, 1.0, DROPOUT)),  # A8: 3 -> 2
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("note_id", _m(1, 1.0, DROPOUT)),  # A9: 2 -> 1 (halves per-note deploy state)
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
