# TRACK-2 A16 (2026-07-25), THE GOAL-LINE CUT: d_model 96 -> 64 (N_HEADS 3 -> 2, head
# dim K=32 kept), built on the A15 champion arch (which is A14's halved LoRAs at d=96).
# Depth floors unchanged (card 2L, deck 4L, note 1L, preset 3L, user 3L), 9 mixer strips,
# LoRAs at the A14 halving (decay/a/gate 8, v0-mix 4).
#
# 388,032 params = -52.0% vs A15's 808,762 = **7.11x below the original 2.76M**, i.e. past
# Andrew's >=5x target in one cut. Per-card state 6,528 -> 4,352 floats (-33%).
# Gate: ratio vs A15 on the val half -- dparams 420,730 => allowed +0.000421/mode.
#
# Rationale for jumping straight to 64 rather than an intermediate: A15 showed the trunk
# had real width slack (its 41% cut spent only 41%/64% of the bar), and the ratio gate
# scales the allowance with the params bought. If A16 misses, the retry ladder is
# d_model 80 (5 heads x K=16 -- the WKV kernel is K-dynamic and the track-1 champion runs
# K=16) and then trimming the SRS heads / input FC, untouched by every cut so far.
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 2  # A16: d_model = 32*2 = 64  (A15 was 3 -> 96, A14 4 -> 128)

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
    ("card_id", _m(2, 1.0, DROPOUT)),      # A8: 3 -> 2
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("note_id", _m(1, 1.0, DROPOUT)),      # A9: 2 -> 1 (halves per-note deploy state)
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
