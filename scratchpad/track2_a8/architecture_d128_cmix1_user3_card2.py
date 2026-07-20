# TRACK-2 A8 (2026-07-21), based on the A7 arch: the A1/A6 d=128 cmix-1.0 arch with the USER stream cut 4L -> 3L.
# Base = track2_a1/architecture_d128_cmix1.py (see its header). The layer cut removes user.L3
# entirely (time_mixer 82,952 + channel_mixer 33,152 = 116,104 params); combined in the A7
# bundle with RWKV_STRIP_CMIX additions note_id:1 + deck_id:2 (the next-tier mixer strips).
# Rationale: user = the consistently lowest-saliency stream across all 4 valid grad
# recordings (its L1/L2 mixers already stripped in A6; its L3 mixer tops the next tier;
# its L2/L3 time-mixers rank bottom-tier among time mixers).
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
    ("card_id", _m(2, 1.0, DROPOUT)),  # A8: 3 -> 2 (both card.L2 units bottom-tier; shrinks per-card deploy state)
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("note_id", _m(2, 1.0, DROPOUT)),
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
