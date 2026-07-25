# TRACK-2 A17 (2026-07-26), the intermediate WIDTH rung after A16 found the floor:
# d_model 96 -> 80 via 5 heads x K=16 (A15 was 3 x 32; A16's 2 x 32 = 64 REJECTED at
# ~1.8x the bar in both modes). K=16 is proven -- the track-1 champion has run 2 x 16
# since the H2K16 acceptance, and the WKV kernel is K-dynamic for any K dividing 32.
#
# 584,766 params = -27.7% vs A15's 808,762 = 4.72x below the original 2.76M.
# Gate: ratio vs A15 on the val half -- dparams 223,996 => allowed +0.000224/mode
# (tighter than A15's 0.000572 and A16's 0.000421, so this rung must be genuinely cheap).
# BONUS the gate does not score: per-layer state 3,264 -> 1,440 floats (K=16 shrinks the
# WKV matrix H*K*K from 3,072 to 1,280), i.e. per-card state 6,528 -> 2,880 (-56%).
#
# Why 80 and not 5x -- the mult cut (head_fc_mult/features_fc_mult 4->2) would reach
# 529,246 = 5.21x, but the 100-user era rejected exactly that change with imm +0.053, so
# it does not get bundled into a rung that already changes width and head dim.
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
    d_model=HEAD_DIM * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
