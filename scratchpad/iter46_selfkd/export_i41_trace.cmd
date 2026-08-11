@echo off
REM ===========================================================================================
REM Export a fresh RNN parity trace for the ITER-41 CHAMPION (interleaved + _cnd order), into a
REM NEW directory. Required by TRACK2_PORT_PLAN.md gaps 7/8: both existing traces
REM (reference_a18, reference_iter31) are SEQUENTIAL, so neither certifies the interleaved
REM champion -- and a trace is never overwritten, it lands beside the old ones.
REM
REM CPU-only (the RNN path), so it is safe beside the running iter-45 GPU chain.
REM
REM Env = the champion's, from CLAUDE.md's champion block. RWKV_INTERLEAVE=1 and the _cnd arch
REM module are the two pieces the older traces lack.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter46_selfkd\export_i41.log

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
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
set RWKV_CHAMP_CKPT=scratchpad/iter41_ilv/i41_d_10935.pth
set RWKV_CHAMP_SFT=iter41_ilv.safetensors
set RWKV_REF_DIR=reference_iter41

echo ===== EXPORT iter41 trace START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u export_rnn_trace.py >> "%LOG%" 2>&1
echo EXPORT_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
