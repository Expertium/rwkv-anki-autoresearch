# TRACK-2 A1 (2026-07-16): the d=128 arch with ALL channel-mixer factors cut to 1.0.
# Base = architecture_old_d128.py (the A0 anchor arch, 2,762,884 params; mixers at cmf
# [1.5, 2.0, 1.5, 2.0, 2.0] = 972,800 params, 35% of the model). Motivation: the d=32 phase
# found mixer width contributes ~nothing at 5k data (capacity-at-5k family, mixer 1.0==1.5).
# Expected cut ~442k params -> per-100k gate allows <= ~0.00044 degradation per mode vs A0.
# ORIGINAL RWKV architecture (the leaderboard model `RWKV_trained_on_5000_10000.pth`, 2.76M params,
# d=128), transcribed into the CURRENT repo's AnkiRWKVConfig format. The srs-benchmark original
# architecture.py lacks the features_fc_mult/head_fc_mult/num_curves/num_points fields our (modified)
# srs_model.py now expects, so it can't be dropped in directly. Used ONLY to evaluate the OLD
# checkpoint on the same data as the new champion; swap the champion architecture.py back afterward.
# Dims verified against the checkpoint's tensor shapes (features2card 92->512->128, head_fc=512,
# num_curves/points=128, modules [3,4,2,3,4], channel_factor [1.5,2.0,1.5,2.0,2.0], LoRA 16/16/8/16).
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
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("note_id", _m(2, 1.0, DROPOUT)),
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(4, 1.0, DROPOUT_LONG)),
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
