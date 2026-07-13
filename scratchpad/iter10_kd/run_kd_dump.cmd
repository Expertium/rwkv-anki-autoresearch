@echo off
REM ============================================================================
REM iter 10 warmup-KD TEACHER DUMP: the d=128 teacher (pretrain/
REM RWKV_trained_on_101_4999.pth -- never saw eval users 5001-10000) walks the
REM champion 5k batch stream (same db/MAX/seeds -> deterministic composition)
REM for the first KDSTEPS steps under eval-mode no_grad, saving per-step soft
REM targets (p_curve + p_imm_all fp16 + labels checksum) via RWKV_KD_DUMP_OUT
REM mode in train_rwkv. ~1.1 MB/step -> ~0.9 GB at 800.
REM
REM ⚠ ARCH FILE-SWAP: rwkv/architecture.py is REPLACED for the duration and
REM restored after. NEVER run while any other rwkv process may (re)launch --
REM e.g. iter 9's decay/eval phases re-import architecture.py. GPU must be free.
REM
REM Smoke first: edit KDSTEPS=3, run, check dump files; then KDSTEPS=800 rerun
REM (files just overwrite). Launch detached (detach.ps1).
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set KDSTEPS=3
set PY=.venv\Scripts\python.exe
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter10_kd\kd_dump.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
REM the OLD arch defines its own heads/dims -- clear champion overrides; NO QAT
REM (teacher is fp); no compile/trace/vprune; JIT stays ON (plain paths are
REM JIT-clean, ~1.4x on the eager body)
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_SHIFT_PQ=
set RWKV_QAT_PQ_LEARN=
set RWKV_QAT_SHIFT_PQ_LEARN=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_QAT_NORM_BITS=
set RWKV_QAT_FUSED=
set RWKV_QAT_COMPILE=
set RWKV_NO_JIT=
set RWKV_STEP_TRACE=
set RWKV_VPRUNE_REF=
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0
set RWKV_KD_DUMP_OUT=scratchpad/iter10_kd/dump
set RWKV_KD_STEPS=%KDSTEPS%
set RWKV_KD_TEACHER=pretrain/RWKV_trained_on_101_4999.pth

echo ===== KD DUMP START (KDSTEPS=%KDSTEPS%) %DATE% %TIME% ===== > "%LOG%"
copy /y rwkv\architecture.py scratchpad\iter10_kd\architecture_champion_backup.py >> "%LOG%" 2>&1
copy /y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1

%PY% -u -m rwkv.train_rwkv --config scratchpad/iter10_kd/kd_dump_ws.toml >> "%LOG%" 2>&1
set DUMP_EXIT=%ERRORLEVEL%

copy /y scratchpad\iter10_kd\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo [ARCH RESTORED] %DATE% %TIME% >> "%LOG%"
echo DONE_EXIT_%DUMP_EXIT% %DATE% %TIME% >> "%LOG%"
