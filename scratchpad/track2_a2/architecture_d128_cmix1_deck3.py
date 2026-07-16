# TRACK-2 A2 (2026-07-16): A1's arch (all channel mixers 1.0) with the DECK stream cut
# 4 -> 3 layers. Base = architecture_d128_cmix1.py (the A1 champion, 2,320,516 params).
# Target choice: user also has 4L but showed REAL sensitivity at d=32 (iters 6/7: state/depth
# signals + a mode trade), while deck was the insensitive stream (ladder rung: no effect) --
# best prior for a near-free layer cut. Expected cut ~110k params -> per-100k gate allows
# <= ~0.00011 degradation per mode vs A1 (exact params from the run banner / model_stats).
# This run also records RWKV_GRAD_STATS (Andrew 2026-07-16) to rank A3+ targets data-driven.
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
    ("card_id", _m(3, 1.0, DROPOUT)),
    ("deck_id", _m(3, 1.0, DROPOUT_LONG)),  # A2: 4 -> 3
    ("note_id", _m(2, 1.0, DROPOUT)),
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(4, 1.0, DROPOUT_LONG)),
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
