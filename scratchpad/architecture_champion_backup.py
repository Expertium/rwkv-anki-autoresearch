import os
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

N_HEADS = 1  # d=32 (K=C/H=32 kept). Champion arch = iter21 (LoRA 16/16/8/16, [3,3,2,3,3],
# channel_mixer_factor=1.0, WSD + 4-epoch decay). d=32 is capacity-starved -> LoRA RAISED to 16.
# iter29 (CHAMPION): SRS heads 128->64 -> 192,800 params, 12.75 KiB. iter30 (logged, NOT adopted):
# card 3->2 -> 8.5 KiB but imm +0.0011 vs iter0 (budget nearly spent). iter31 (THIS): REBALANCE =
# card 3->2 (state 8.5 KiB) + note 2->3 (UNGATED stream grows to recover imm at NO state cost, since
# only card_id gates state). Goal: 8.5 KiB AND imm headroom preserved. REMINDER: champion = iter29
# (arch_snapshots/arch_iter29.py); restore it if iter31 fails.

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
    # head inner-width multiplier (ahead_head/p_head/w_head = head_fc_mult*d_model). Champion = 4.
    # iter33 tested ALL four (incl. heads) at 2 -> imm CATASTROPHIC (+0.0526); heads MUST stay 4.
    head_fc_mult: int = 4
    # input-encoder (features2card) hidden width = features_fc_mult*d_model. Champion = 4.
    # iter34 tests 2 (cut ONLY the input FC, keep imm-critical heads at 4): ~-8k params, zero state.
    features_fc_mult: int = 4


_layers = [
    (
        "card_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=1,  # iter35: card 2->1 -> per-card state 8.5->4.25 KiB (toward the 1 KB target)
            layer_offset=0,
            total_layers=1,
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
            n_layers=4,  # iter36: deck 3->4 (CHEAP stream, ~few decks/user) compensates card 2->1
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
    (
        "note_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,  # iter36: note back to 3 (note is SEMI-EXPENSIVE deploy; compensate via deck)
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

# ---- State-QAT scope parsing (set RWKV_NO_JIT=1 too). Mirrors the Rust deploy env vars. ----
_QMAX = {"int8": 127.0, "int4": 7.0, "int2": 1.0, "fp32": float("inf")}
_QAT_NAME = {"card": "card_id", "deck": "deck_id", "note": "note_id",
             "preset": "preset_id", "user": "user_id"}
# RWKV_QAT_SCOPE="card:int2,note:int2": per-step int-N fake-quant of each named stream's WKV state.
_qat_scope = os.environ.get("RWKV_QAT_SCOPE", "").strip()
if _qat_scope:
    _qat = {}
    for _entry in _qat_scope.split(","):
        _n, _, _lvl = _entry.strip().partition(":")
        _qat[_QAT_NAME[_n]] = _QMAX[_lvl]
    for _name, _cfg in _layers:
        if _name in _qat:
            _cfg.state_qmax = _qat[_name]
    print("[QAT] state_qmax set: " +
          ", ".join(f"{n}={c.state_qmax}" for n, c in _layers if c.state_qmax != float("inf")))
# RWKV_QAT_LOWRANK_SCOPE="card:2:int4,note:2:int4": per-step rank-r truncation (+ int-N factor quant)
# of each named stream's WKV state -- the low-rank deploy analog. Takes precedence over int-N quant.
_lr_scope = os.environ.get("RWKV_QAT_LOWRANK_SCOPE", "").strip()
if _lr_scope:
    _lr = {}
    for _entry in _lr_scope.split(","):
        _parts = _entry.strip().split(":")
        _lr[_QAT_NAME[_parts[0]]] = (int(_parts[1]), _QMAX[_parts[2]] if len(_parts) > 2 else float("inf"))
    for _name, _cfg in _layers:
        if _name in _lr:
            _cfg.state_lowrank_rank, _cfg.state_lowrank_fqmax = _lr[_name]
    print("[QAT-LOWRANK] set: " +
          ", ".join(f"{n}=rank{c.state_lowrank_rank}/fq{c.state_lowrank_fqmax}"
                    for n, c in _layers if c.state_lowrank_rank > 0))

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT,
    num_curves=64, num_points=64,  # iter29: halve SRS-head resolution (champion=128/128)
)  # features_fc_mult/head_fc_mult default to 4 (both REQUIRED -- iter33/34 showed cutting either fails imm)
