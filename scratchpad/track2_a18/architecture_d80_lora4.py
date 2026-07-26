# TRACK-2 A18 (2026-07-26): A17's width (d_model 80 = 5 heads x K=16) PLUS the second
# LoRA halving (decay/a/gate 8->4, v0-mix 4->2). 557,246 params = 4.95x below the original
# 2.76M -- effectively Andrew's >=5x target.
#
# WHY THIS EXACT BUNDLE (conduct rule: a near-miss gets an in-family retry, not a
# writeoff). A17 missed by 26 MILLIONTHS: ahead +0.000250 vs an allowance of 0.000224
# (112% of bar) while imm passed at 83%. That margin is an order of magnitude inside the
# ~0.0004 cross-seed spread, so it is noise-limited, not a real floor. Halving the LoRAs
# again buys 27,520 more params of allowance (0.000224 -> 0.000252/mode), which A17's
# MEASURED cost would clear at 99% (ahead) / 74% (imm) -- and A14 showed the first LoRA
# halving was free-to-positive at d=128 (it IMPROVED both modes), so the added cut has a
# good prior. If A18 also lands ~112% of bar, the honest read is that d=80 is the floor.
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

HEAD_DIM = 16  # A17: K 32 -> 16 (kernel is K-dynamic; the track-1 champion runs K=16)

N_HEADS = 5  # A17: d_model = 16*5 = 80 (A15 was 3 x 32 = 96)

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
        d_model=HEAD_DIM * N_HEADS, n_heads=N_HEADS, n_layers=n_layers,
        layer_offset=0, total_layers=n_layers, channel_mixer_factor=cmf,
        decay_lora=4, a_lora=4, v0_mix_amt_lora=2, gate_lora=4,  # A18: halved AGAIN
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
    d_model=HEAD_DIM * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
