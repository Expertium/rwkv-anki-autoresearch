@echo off
REM Export RNN traces for users 101-160 into reference_iter45/, so the Rust engine can dump a
REM SHIFT CORPUS large enough to fit a 4096-entry (12-bit) PQ catalog.
REM
REM WHY: the 3 parity users give only ~10k unit vectors per role -- ~2 per centroid at ncent=4096,
REM which fits noise rather than the state distribution. 60 users gives ~60-70k per role (~15-17
REM per centroid), which is thin but usable, and BOTH candidate catalogs (m=2 and m=5) are fitted
REM on the SAME corpus so the comparison between them stays fair either way.
REM
REM CPU-only and thread-limited: iter 46 holds the GPU and its fetch workers need CPU headroom.
REM Training is seeded, so co-tenant CPU load costs wall-clock, never correctness.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\qat_tax\export_range.log

set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=3
set MKL_NUM_THREADS=3
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

echo ===== EXPORT traces 101-160 START %DATE% %TIME% ===== > "%LOG%"
.venv\Scripts\python.exe -u scratchpad/export_traces_range.py 101 161 >> "%LOG%" 2>&1
echo RANGE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
