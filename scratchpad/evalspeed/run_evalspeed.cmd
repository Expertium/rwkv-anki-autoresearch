@echo off
REM Eval-phase A/B: SEQUENTIAL (what every runner does today: --shards 1 --solo-threshold 0) vs the
REM tool's own PHASED mode (giants solo first, the rest LPT-split across 2 shards). Users 5001-5400.
REM
REM WHY. The eval is ~4.2 h of a ~10.6 h iteration -- 40% of every experiment's wall clock -- and the
REM LIVE RULE "d=80 evals UNSHARDED" dates from the d=128/d=80 memory scare, before the solo phase
REM existed to keep mega-users off the parallel shards. eval_sharded.py's own docstring predicts
REM ~0.56*W (~1.8x). If it holds, every future iteration is ~1.8 h cheaper and the endgame's arms are
REM cheaper too. Measured, not assumed -- and on a subset, so a WDDM wedge costs minutes not a gate.
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONIOENCODING=utf-8
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\evalspeed
set LOG=%DIR%\evalspeed.log
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_ID_FEATURES=1
set RWKV_REAL_CYCLES=1
set RWKV_ZERO_FEATURES=
set RWKV_EVAL_PAVA=1
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_EVAL_SHARD_DIR=%DIR%\shards
echo ===== EVAL SPEED A/B START %DATE% %TIME% ===== > "%LOG%"
echo --- ARM 1: sequential (today's setting) %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval_seq.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 >> "%LOG%" 2>&1
echo --- ARM 1 done rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo --- ARM 2: phased (solo giants, then 2 shards) %TIME% >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval_shd.toml --shards 2 --solo-threshold 1000000 --fetch-per-shard 2 --threads-per-shard 3 >> "%LOG%" 2>&1
echo --- ARM 2 done rc %ERRORLEVEL% %TIME% >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
