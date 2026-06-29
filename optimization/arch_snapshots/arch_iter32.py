from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 1  # d=32 (K=C/H=32 kept). Champion arch = iter21 (LoRA 16/16/8/16, [3,3,2,3,3],
# channel_mixer_factor=1.0, WSD + 4-epoch decay). d=32 is capacity-starved -> LoRA RAISED to 16.
# CHAMPION = iter31: card 3->2 + note 2->3 ([2,3,3,3,3]) -> 8.5 KiB state, 192,800 params, imm 0.315438.
# iter32 (THIS): same card 3->2 + state 8.5 KiB, but grow USER 3->4 instead of note ([2,3,2,3,4],
# param-neutral 192,800). Tests whether the head-adjacent user stream recovers the card-3->2 imm BETTER
# than note did (iter31 0.315438) -> a strictly-better 8.5 KiB champion. REMINDER: champion = iter31
# (arch_snapshots/arch_iter31.py); restore it if iter32 is not better.

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
            n_layers=2,  # iter32: note reverted to 2 (test growing USER instead of note)
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
            n_layers=4,  # iter32: user 3->4 (grow the head-adjacent UNGATED stream to recover card-3->2 imm)
            layer_offset=0,
            total_layers=4,
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
)
