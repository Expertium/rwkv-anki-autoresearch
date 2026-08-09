#!/bin/bash
# Local Linux WS phase: iter41 champion recipe (interleave + fine-to-coarse order), MINUS
# distillation (RWKV_KD_MIX/RWKV_KD_ALPHA dropped -- the Windows-only teacher dump
# C:\rwkv_kd_dump\t128_seedpair_65k isn't available on this machine). Everything else matches
# scratchpad/iter41_ilv/run_iter41.cmd's WS block exactly.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=7
export RWKV_DETERMINISTIC=1
export RWKV_AUGMENT_SEED=4321
export RWKV_EMPTY_CACHE_EVERY=1
export RWKV_EMPTY_CACHE_WINDOW=0
export RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
export RWKV_GRU_HEAD=3
export RWKV_PAVA_LAMBDA=0.2
export RWKV_PROBE_DENSITY=0.08
export RWKV_PROBE_DUR=0.0
export RWKV_MUON=1
export RWKV_MUON_LR=0.0025
export RWKV_MUON_MOMENTUM=0.95
export RWKV_NO_AHEAD_RESIDUAL=1
export RWKV_STRIP_L0_VLORA=1
export RWKV_ZERO_FEATURES=22
export RWKV_STATE_CLAMP_TAU=300
export RWKV_STATE_CLAMP_WINDOW=32768
export RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
export RWKV_WEIGHT_DECAY=0.01
export RWKV_WEIGHT_DECAY_HEAD=0.01
export RWKV_CLIP=0.25
export RWKV_ADAMW_BETA2=0.999
export RWKV_DROPOUT_SCALE=0.5
export RWKV_GRAD_STATS=scratchpad/local_run/grad_stats.json
# ---- THE LEVER ----
export RWKV_INTERLEAVE=1
# the adopted speed stack (training only; cleared before eval)
export RWKV_MUON_BATCHED=1
export RWKV_NO_JIT=1
export RWKV_QAT_COMPILE=1
# NOTE: no RWKV_KD_MIX / RWKV_KD_ALPHA -- see header.

echo "=== WS start $(date) ==="
.venv/bin/python -u -m rwkv.train_rwkv --config scratchpad/local_run/local_ws.toml
echo "=== WS done $(date) ==="
