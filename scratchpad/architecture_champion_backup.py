import os
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

# Head config: d_model = N_HEADS * HEAD_DIM (= K). CHAMPION = N_HEADS=1, HEAD_DIM=32 -> d_model=32, K=32.
# RWKV_N_HEADS / RWKV_HEAD_DIM env overrides (2026-06-30) enable the 2x-smaller-state arch H=2/K=16
# (d_model=32, K=16) now that the CUDA kernel supports K<32 (K must DIVIDE 32; parity verified via
# scratchpad/test_k16_wkv.py). Default unset == champion (byte-identical). WKV per-layer state =
# N_HEADS*HEAD_DIM^2 floats: H1/K32 = 1024, H2/K16 = 512 (HALF). (Lineage: iter21/iter29 -> tuned d=32 ->
# 1500-user data champion; arch_snapshots/arch_iter29.py is the old reference.)
N_HEADS = int(os.environ.get("RWKV_N_HEADS") or "1")
HEAD_DIM = int(os.environ.get("RWKV_HEAD_DIM") or "32")

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
            d_model=N_HEADS * HEAD_DIM,
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
            d_model=N_HEADS * HEAD_DIM,
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
            d_model=N_HEADS * HEAD_DIM,
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
            d_model=N_HEADS * HEAD_DIM,
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
            d_model=N_HEADS * HEAD_DIM,
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

# Capacity env overrides (research-phase arch levers, 2026-06-30): default unset = champion values.
# RWKV_CHANNEL_MIXER_FACTOR (per-block FFN width mult; champion 1.0) and RWKV_LORA (decay/a/gate LoRA rank;
# champion 16) add params with ZERO per-entity state cost. Arch-agnostic -- mutate whatever streams exist.
_cmf_env = os.environ.get("RWKV_CHANNEL_MIXER_FACTOR")
if _cmf_env:
    for _n, _c in _layers:
        _c.channel_mixer_factor = float(_cmf_env)
_lora_env = os.environ.get("RWKV_LORA")
if _lora_env:
    _lv = int(_lora_env)
    for _n, _c in _layers:
        _c.decay_lora = _lv
        _c.a_lora = _lv
        _c.gate_lora = _lv

# ---- State-QAT scope parsing (set RWKV_NO_JIT=1 too). Mirrors the Rust deploy env vars. ----
_QMAX = {"int8": 127.0, "int4": 7.0, "int3": 3.0, "int2": 1.0, "fp32": float("inf")}
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
# RWKV_QAT_SHIFT_SCOPE="card:int3,note:int3": per-step int-N fake-quant of each named stream's token-shift
# vectors -- the QAT analog of the deploy RWKV_QUANT_SHIFTS + RWKV_STATE_SHIFT_LEVEL. Independent of the WKV
# factor level, so shifts can be trained robust to a COARSER bit-width than the WKV (e.g. WKV int4 + shift int3).
_shift_scope = os.environ.get("RWKV_QAT_SHIFT_SCOPE", "").strip()
if _shift_scope:
    _sh = {}
    for _entry in _shift_scope.split(","):
        _n, _, _lvl = _entry.strip().partition(":")
        _sh[_QAT_NAME[_n]] = _QMAX[_lvl]
    for _name, _cfg in _layers:
        if _name in _sh:
            _cfg.state_shift_qmax = _sh[_name]
    print("[QAT-SHIFT] state_shift_qmax set: " +
          ", ".join(f"{n}={c.state_shift_qmax}" for n, c in _layers if c.state_shift_qmax != float("inf")))

# SRS-head resolution env overrides (research-phase arch lever, 2026-06-30): default 64 = champion (iter29
# halved 128->64 for params). Set RWKV_NUM_CURVES / RWKV_NUM_POINTS to sweep (e.g. 128) -- pure params, ZERO
# state cost; the Rust engine auto-derives these from weight shapes. Both train AND eval must set the same value.
_num_curves = int(os.environ.get("RWKV_NUM_CURVES") or "64")
_num_points = int(os.environ.get("RWKV_NUM_POINTS") or "64")
DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=N_HEADS * HEAD_DIM, modules=_layers, dropout=DROPOUT,
    num_curves=_num_curves, num_points=_num_points,
)  # features_fc_mult/head_fc_mult default to 4 (both REQUIRED -- iter33/34 showed cutting either fails imm)
