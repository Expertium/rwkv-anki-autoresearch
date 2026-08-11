@echo off
REM Export the ITER-45 CHAMPION's weights + a fresh RNN parity trace into reference_iter45/.
REM Two jobs at once:
REM   1. the safetensors the Rust engine needs to dump a SHIFT CORPUS at C=80 (the q72u shift
REM      codebook is C=32-shaped and hard-fails on this model -- verified 2026-08-12);
REM   2. the pre-ship parity trace for the new champion (reference_iter41 certified iter 41, not 45).
REM CPU-only, so it can run while iter 46 holds the GPU.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\export_i45.log

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=4
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_CHAMP_CKPT=scratchpad/iter45_kddecay/i45_d_10935.pth
set RWKV_CHAMP_SFT=iter45_kddecay.safetensors
set RWKV_REF_DIR=reference_iter45

echo ===== EXPORT iter45 START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u export_rnn_trace.py >> "%LOG%" 2>&1
echo EXPORT_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
