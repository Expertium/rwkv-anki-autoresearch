@echo off
REM ===========================================================================================
REM CELL B OF THE INTERVAL DECOMPOSITION: train on end-to-END, evaluate on end-to-START.
REM
REM This is featA2's OWN decay checkpoint, unchanged, scored against the end-to-start test db.
REM No training. One eval, about 2.3 h.
REM
REM WHY IT EXISTS. The interval question has THREE cells, not two -- the same shape as the
REM 2026-08-12 QAT-tax decomposition:
REM
REM   A  train e2e, eval e2e  = featA2. What we report today. Deploy CANNOT reproduce it.
REM   B  train e2e, eval e2s  = THIS RUN. What the SHIPPED model actually achieves in Anki.
REM   C  train e2s, eval e2s  = the e2s arm. What a matched model would achieve.
REM
REM   B minus A = the cost of the train/deploy mismatch AS CURRENTLY SHIPPED.
REM   C minus B = what retraining on end-to-start recovers.
REM   C minus A = the honest total correction to the published number.
REM
REM Without B, A-versus-C conflates those two effects and no number describes the model that
REM actually runs. A live Anki scheduler computes now() minus last_review_time -- end-to-start,
REM structurally, because duration(k) has not happened yet (jschoreels fork, rwkv.rs line 322).
REM Cell B is therefore not hypothetical; it is the production configuration.
REM
REM SINGLE VARIABLE vs featA2's own eval: only the eval db changes. Same checkpoint, same env,
REM same user range, same rectified metric, same label_filter_db, same review count (verified:
REM test_db_5k_fix and test_db_5k_e2s both hold 170,384 entries).
REM
REM Outputs go to a SEPARATE directory so featA2's own eval.toml and logs are not overwritten.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set TAG=featA2xe2s
set CKPTDIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featA2
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featA2_on_e2s
set LOG=%DIR%\featA2xe2s.log
set STAMP=%RANDOM%%RANDOM%
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7

REM ---- env copied from run_featA2_evalonly.cmd, unchanged except the two db vars ----
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_ID_FEATURES=
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_LABEL_FILTER_DB=label_filter_db
REM ---- THE ONLY CHANGE FROM CELL A ----
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_e2s
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_e2s

if not exist "%DIR%" mkdir "%DIR%"
echo ===== FEATA2-ON-E2S (CELL B) START %DATE% %TIME% ===== > "%LOG%"

if not exist "%CKPTDIR%\featA2_d_10935.pth" (
  echo %TAG% DECAY_CKPT_MISSING %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_31 %DATE% %TIME% >> "%LOG%"
  exit /b 31
)

set RWKV_EVAL_PAVA=1
.venv\Scripts\python.exe scratchpad/write_eval_toml.py "%CKPTDIR%" featA2_d %DIR%\eval.toml RWKV-%TAG% RWKV-P-%TAG% 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% ETOML_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 %DATE% %TIME% >> "%LOG%"
  exit /b 24
)

REM ---- guard 1: it must score the END-TO-START db, or this cell is a duplicate of cell A ----
findstr /C:"F:/rwkv_lmdb/test_db_5k_e2s" %DIR%\eval.toml >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_DB_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_34 %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
REM ---- guard 2: it must score FEATA2's checkpoint. The wrong-checkpoint trap cost iter 47. ----
findstr /C:"featA2_d_10935.pth" %DIR%\eval.toml >nul
if not %ERRORLEVEL%==0 (
  echo %TAG% WRONG_CKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_32 %DATE% %TIME% >> "%LOG%"
  exit /b 32
)

if exist "result\RWKV-%TAG%.jsonl" del /q "result\RWKV-%TAG%.jsonl"
if exist "result\RWKV-P-%TAG%.jsonl" del /q "result\RWKV-P-%TAG%.jsonl"
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo %TAG% EVAL_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo %TAG% EVAL_OK %TIME% >> "%LOG%"

REM Terminal marker BEFORE endlocal: endlocal restores the pre-setlocal environment, so the
REM LOG variable would expand to empty and the marker would be written to "".
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
