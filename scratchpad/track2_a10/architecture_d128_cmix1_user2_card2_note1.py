# TRACK-2 A10 (2026-07-22), based on the A9 arch: the d=128 cmix-1.0 arch with card 2L,
# note 1L, and now the USER stream cut 3L -> 2L. Base = track2_a9/
# architecture_d128_cmix1_user3_card2_note1.py (see its header chain). The layer cut
# removes user.L2 (time_mixer 82,952 + its already-dummy channel mixer) -- user.L1/L2
# time-mixers ranked #1/#4 lowest saliency in A9's WS grad recording; this is user
# depth's THIRD consecutive prunable verdict (4L->3L in A7, now 3L->2L).
# Bundled in the A10 cmd with RWKV_STRIP_CMIX additions note_id:0 (the last note mixer,
# kept in A9 for caution -- now #5-lowest and A9 ran the cleanest of the chain) +
# deck_id:3 (deck.L3.channel_mixer #3-lowest; a MIXER strip, not the A2 depth cut).
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
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(2, 1.0, DROPOUT_LONG)),  # A10: 3 -> 2 (A7: 4 -> 3)
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
