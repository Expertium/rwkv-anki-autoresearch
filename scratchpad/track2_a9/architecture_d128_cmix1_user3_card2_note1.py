# TRACK-2 A9 (2026-07-21), based on the A8 arch: the d=128 cmix-1.0 arch with user 3L,
# card 2L, and now the NOTE stream cut 2L -> 1L. Base = track2_a8/
# architecture_d128_cmix1_user3_card2.py (see its header chain). The layer cut removes
# note.L1 entirely (time_mixer 82,952 + its already-dummy channel mixer) -- note.L1.time_mixer
# ranked #2-lowest saliency in A8's WS grad recording (6.9e-07 mean|g*w|), and a 1-layer
# note stream HALVES per-note d=128 deploy state (note state dominates deploy memory).
# Bundled in the A9 cmd with RWKV_STRIP_CMIX additions user_id:0 + preset_id:0 (the #1 and
# #6 lowest-saliency units; note_id:1 leaves the strip list with the layer). Note.L0's own
# channel mixer is deliberately KEPT (the only mixer left in the stream; stability caution
# after A8's watch item).
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
    ("note_id", _m(1, 1.0, DROPOUT)),  # A9: 2 -> 1 (halves per-note deploy state)
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
