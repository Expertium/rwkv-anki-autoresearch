from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 1  # d=32 (K=C/H=32 kept). Champion arch = iter21 (LoRA 16/16/8/16, [3,3,2,3,3],
# channel_mixer_factor=1.0, WSD + 4-epoch decay). d=32 is capacity-starved -> LoRA RAISED to 16.
# CHAMPION = iter31: [2,3,3,3,3], SRS heads 64, 8.5 KiB state, 192,800 params, imm 0.315438.
# iter33 (THIS): iter31 + FC/head inner width halved 4->2 (head_fc_mult, below) -> 169,888 params
# (16.3x vs iter0, -11.9% vs iter31), state UNCHANGED 8.5 KiB (FC/heads not recurrent). Pure param cut;
# like the SRS-head cut, the 4*d_model FC width is likely over-provisioned at d=32. REMINDER: champion =
# iter31 (arch_snapshots/arch_iter31.py, head_fc_mult=4); restore it if iter33 fails the gate.

DROPOUT = 0.02
DROPOUT_LONG = 0.05
DROPOUT_LAYER = 0.01


@dataclass
class AnkiRWKVConfig:
    d_model: int
    modules: list
    dropout: float
    # SRS-head resolution (param-reduction lever; pure params, zero RNN-state cost):
    # num_curves = # basis forgetting-curves in the softmax mixture (drives imm/RWKV-P);
    # num_points = # sample points the ahead head interpolates over (drives ahead mode).
    # Baseline/champion = 128/128. iter29 tests 64/64 (-16,384 params, ~7.8%).
    num_curves: int = 128
    num_points: int = 128
    # FC/head inner-width multiplier: features_fc_dim / ahead_head_dim / p_head_dim / w_head_dim
    # = head_fc_mult * d_model. Baseline = 4. iter33 tests 2 (~-22k params, zero state cost).
    head_fc_mult: int = 4


_layers = [
    (
        "card_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=2,  # iter30: card stream 3->2 layers -> per-card state 12.75->8.5 KiB
            layer_offset=0,
            total_layers=2,
            channel_mixer_factor=1.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "deck_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=1.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "note_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,  # iter31: note 2->3 (UNGATED stream grows to recover the card-3->2 imm cost)
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=1.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "preset_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=1.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "user_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=1.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=64, num_points=64,  # iter29: halve SRS-head resolution (champion=128/128)
    head_fc_mult=2,  # iter33: halve FC/head inner width 4->2 (champion=4)
)
