# TRACK-2 A14 (2026-07-23), based on the A9/A13 champion arch: the d=128 cmix-1.0 arch
# (card 2L, deck 4L, note 1L, preset 3L, user 3L -- all depth floors) with ALL FOUR
# LoRA dims HALVED: decay 16->8, a 16->8, gate 16->8, v0-mix 8->4, every stream.
# First STRUCTURAL cut after the depth ladder closed (A12) -- the LoRA projections are
# a distributed ~6% param mass the per-unit saliency ranking cannot target. ~-88k vs
# A13's 1,468,724. Recipe = A13's (incl. RWKV_ZERO_FEATURES=22); gate = ratio vs A13
# on the val half.
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 4  # d_model = 32*4 = 128

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
