# ITER 41 (2026-08-09): architecture_d80_lora4.py with the stream order corrected to TRUE
# FINE-TO-COARSE -- card -> NOTE -> DECK -> preset -> user (Andrew's directive after the
# order audit: every historical arch file ran card -> DECK -> NOTE, which is not monotone
# in granularity -- notes ~0.9x cards, decks ~56/user. CLAUDE.md's invariant text had
# always CLAIMED card->note->deck; the code never did it until this file).
#
# ONLY the tuple order changes. Each stream keeps its own depth/mixer/dropout (they travel
# with the name): card 2L, note 1L, deck 4L, preset 3L, user 3L. Param count identical.
# prepare() builds gathers by iterating these tuples, so training, eval and the RNN deploy
# path (state routing is BY NAME since the iter-41 refactor) all reorder coherently; the KD
# dump stays valid (its checksum covers labels, which do not depend on stream order).
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

HEAD_DIM = 16  # A17: K 32 -> 16 (kernel is K-dynamic; the track-1 champion runs K=16)

N_HEADS = 5  # A17: d_model = 16*5 = 80 (A15 was 3 x 32 = 96)

# See architecture_d80_lora4.py for the RWKV_DROPOUT_SCALE restoration note (2026-08-03).
import os

_DROPOUT_SCALE = float(os.environ.get("RWKV_DROPOUT_SCALE") or "1")
DROPOUT = 0.02 * _DROPOUT_SCALE
DROPOUT_LONG = 0.05 * _DROPOUT_SCALE
DROPOUT_LAYER = 0.01 * _DROPOUT_SCALE


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
    ("note_id", _m(1, 1.0, DROPOUT)),  # A9: 2 -> 1; MOVED before deck (fine-to-coarse)
    ("deck_id", _m(4, 1.0, DROPOUT_LONG)),
    ("preset_id", _m(3, 1.0, DROPOUT_LONG)),
    ("user_id", _m(3, 1.0, DROPOUT_LONG)),  # A7: 4 -> 3
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=HEAD_DIM * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=128, num_points=128,
)
