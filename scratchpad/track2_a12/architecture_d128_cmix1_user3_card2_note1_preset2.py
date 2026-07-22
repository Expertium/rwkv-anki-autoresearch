# TRACK-2 A12 (2026-07-22), based on the A9 CHAMPION arch (user 3L -- A10/A11's user-2L
# was REJECTED, user depth floors at 3): the d=128 cmix-1.0 arch with card 2L, note 1L,
# user 3L, and now the PRESET stream cut 3L -> 2L. Base = track2_a9/
# architecture_d128_cmix1_user3_card2_note1.py (see its header chain). The layer cut
# removes preset.L2 (time_mixer 82,952 + its already-dummy channel mixer) -- preset.L1/L2
# time-mixers ranked #6/#7 lowest saliency in A9's WS grad recording; preset is the ONE
# stream whose depth cut is still untried (floors mapped: card=2, note=1, user=3, deck=4).
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
        decay_lora=16, a_lora=16, v0_mix_amt_lora=8, gate_lora=16,
        dropout=dropout, dropout_layer=DROPOUT_LAYER,
    )


_layers = [
    ("card_id", _m(2, 1.0, DROPOUT)),  # A8: 3 -> 2
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("note_id", _m(1, 1.0, DROPOUT)),  # A9: 2 -> 1
    ("preset_id", _m(2, 1.0, DROPOUT_LONG)),  # A12: 3 -> 2
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3 (floors at 3)
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
