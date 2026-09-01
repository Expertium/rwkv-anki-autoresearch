@echo off
REM ===========================================================================================
REM THE FIXC CONTROL ARM -- END-TO-END intervals, everything else identical to the e2s
REM re-base. Paired with it, this isolates the interval definition exactly.
REM (generated from run_e2s_rebase.cmd by mk_fixc_arm.py; do not hand-edit)
REM ORIGINAL HEADER FOLLOWS.
REM THE E2S CHAMPION RE-BASE. Andrew 2026-08-30, verbatim: "e2s should be used both in train AND
REM eval. That should be the new default for all future runs."
REM
REM Every number in the record -- iter 53 included -- is end-to-END, so an e2s run is not
REM comparable to any of them. This run establishes the NEW BASELINE that future candidates get
REM gated against. It is iter 53's recipe with three db paths changed and nothing else.
REM
REM ---- WHY PHASE 1 EXISTS, AND WHY SKIPPING IT WOULD BE SILENT ----
REM The existing dump C:\rwkv_kd_dump\t128_seedpair_65k holds teacher logits computed on
REM end-to-END inputs. Its ONLY identity check is a per-step labels_sum, and labels are RATINGS,
REM which the interval definition does not touch. A student on e2s dbs would therefore PASS the
REM checksum while distilling toward predictions for different inputs. Identical in shape to the
REM augmentation/KD incompatibility already on record: the checksum proves LABEL alignment and
REM gets read as proving BATCH alignment. So the dump is regenerated on the e2s batch stream.
REM The dump lands on F: -- C: has 54 GB free and a dump is 7.1 GB of 10,935 small files.
REM
REM PHASES: 0 smoke dump (5 steps, seconds, catches a config error before 1.5 h)
REM         1 full teacher dump, 10,935 steps, forward-only
REM         2 WS, KD alpha 0.9      3 decay, KD alpha 0.5      4 rectified VAL-half eval
REM
REM The teacher runs with its OWN arch and none of the trunk flags, in its own setlocal scope --
REM teacher and student cannot share a process. Probe vars are DATA-side and must agree.
REM
REM Do NOT edit while running: cmd.exe re-reads a batch file from a saved byte offset.
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
setlocal
cd /d C:\Users\Andrew\rwkv-anki-autoresearch

set TAG=fixc
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\fixc_arm
set LOG=%DIR%\fixc_arm.log
set STAMP=%RANDOM%%RANDOM%
set WSSTEPS=10935
set DUMP=F:\rwkv_kd_dump\t128_fixc_65k
set SMOKE=F:\rwkv_kd_dump\t128_fixc_smoke

if not exist "%DIR%" mkdir "%DIR%"
echo ===== FIXC ARM START %DATE% %TIME% ===== > "%LOG%"

REM ---- phase 0b: the control and treatment dbs must differ ONLY in the interval.
REM Same rows, same entity ids, DIFFERENT features. The third condition is the one that
REM matters: if the e2s lever had leaked into the control's build, conditions 1 and 2
REM would both pass and the experiment would measure nothing while reporting a clean null.
.venv\Scripts\python.exe scratchpad/features_rebuild/assert_pair_single_variable.py F:/rwkv_lmdb/train_db_5k_h1_fixc F:/rwkv_lmdb/train_db_5k_h1_e2s 40 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo PAIR_INVALID -- refusing to run a control that is not one >> "%LOG%"
  echo DONE_EXIT_47 %DATE% %TIME% >> "%LOG%"
  exit /b 47
)
echo PHASE 0b OK (single-variable pair) %TIME% >> "%LOG%"

REM =========================== TEACHER DUMP (d=128, forward only) ============================
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
REM The teacher is the ORIGINAL model: its own arch, none of the trunk flags. Cleared
REM explicitly, because this .cmd may run in a shell where they linger.
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py
set RWKV_GRU_HEAD=
set RWKV_PAVA_LAMBDA=
set RWKV_MUON=
set RWKV_MUON_LR=
set RWKV_MUON_MOMENTUM=
set RWKV_MUON_BATCHED=
set RWKV_MUON_INCLUDE_LORA=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_DROPOUT_SCALE=
set RWKV_NO_AHEAD_RESIDUAL=
set RWKV_STRIP_L0_VLORA=
set RWKV_ZERO_FEATURES=
set RWKV_STATE_CLAMP_TAU=
set RWKV_STATE_CLAMP_WINDOW=
set RWKV_STRIP_CMIX=
set RWKV_INTERLEAVE=
set RWKV_VPRUNE_REF=
set RWKV_STEP_TRACE=
set RWKV_GRAD_STATS=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
REM Probe vars are DATA-side: teacher and student must agree on row layout.
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_KD_TEACHER=pretrain/RWKV_trained_on_101_4999.pth

echo --- PHASE 0: smoke dump, 5 steps %TIME% >> "%LOG%"
if exist "%SMOKE%" rmdir /s /q "%SMOKE%"
mkdir "%SMOKE%" 2>nul
set RWKV_KD_DUMP_OUT=%SMOKE%
set RWKV_KD_STEPS=5
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/fixc_arm/dump.toml > "%DIR%\smoke_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo SMOKEFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_20 %DATE% %TIME% >> "%LOG%"
  exit /b 20
)
if not exist "%SMOKE%\step_5.pt" (
  echo SMOKE_NO_ARTIFACT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_26 %DATE% %TIME% >> "%LOG%"
  exit /b 26
)
echo PHASE 0 OK %TIME% >> "%LOG%"

echo --- PHASE 1: full teacher dump, %WSSTEPS% steps %TIME% >> "%LOG%"
if exist "%DUMP%" rmdir /s /q "%DUMP%"
mkdir "%DUMP%" 2>nul
set RWKV_KD_DUMP_OUT=%DUMP%
set RWKV_KD_STEPS=%WSSTEPS%
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/fixc_arm/dump.toml > "%DIR%\dump_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DUMPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_21 %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
REM Gate on the LAST step's artifact, not on exit 0: a short dump makes the student exit 43
REM thousands of steps later, which is an expensive way to learn this.
if not exist "%DUMP%\step_%WSSTEPS%.pt" (
  echo DUMP_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_27 %DATE% %TIME% >> "%LOG%"
  exit /b 27
)
echo PHASE 1 OK %TIME% >> "%LOG%"
endlocal

REM ============================== STUDENT: iter 53's recipe ==================================
setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_MUON_INCLUDE_LORA=1
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_FSRS_CARD=
set RWKV_ID_FEATURES=
REM ---- e2s dbs for decay and eval too. The tomls carry them for WS. ----
set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_fixc
set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_fixc
set RWKV_LABEL_FILTER_DB=label_filter_db

echo --- PHASE 2: WS, KD alpha 0.9 %TIME% >> "%LOG%"
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/fixc_arm/ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_22 %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
if not exist "%DIR%\fixc_ws_%WSSTEPS%.pth" (
  echo WS_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_28 %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
REM The re-base must be the champion ARCHITECTURE, unchanged. A different count means a flag
REM failed to reach the workers and the baseline would be for a model we did not intend.
findstr /C:"Trainable parameters: 558212" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo WRONG_PARAM_COUNT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_33 %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
findstr /C:"alpha FIXED at 0.9" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo WRONG_WS_ALPHA %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_35 %DATE% %TIME% >> "%LOG%"
  exit /b 35
)
echo PHASE 2 OK %TIME% >> "%LOG%"

echo --- PHASE 3: decay, KD alpha 0.5 %TIME% >> "%LOG%"
set RWKV_KD_ALPHA=0.5
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/fixc_arm fixc_ws fixc_d %DIR%\decay.toml F:/rwkv_lmdb/train_db_5k_h1_fixc 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DSETUP_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_23 %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"fixc_ws_%WSSTEPS%" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo WRONGCKPT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_32 %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config %DIR%\decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_24 %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo WRONG_DECAY_ALPHA %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_36 %DATE% %TIME% >> "%LOG%"
  exit /b 36
)
if not exist "%DIR%\fixc_d_%WSSTEPS%.pth" (
  echo DECAY_SHORT %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_29 %DATE% %TIME% >> "%LOG%"
  exit /b 29
)
echo PHASE 3 OK %TIME% >> "%LOG%"

echo --- PHASE 4: rectified VAL-half eval on the e2s test db %TIME% >> "%LOG%"
set RWKV_EVAL_PAVA=1
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/fixc_arm fixc_d %DIR%\eval.toml RWKV-fixc RWKV-P-fixc 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo ETOML_FAILED_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_25 %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
findstr /C:"F:/rwkv_lmdb/test_db_5k_fixc" %DIR%\eval.toml >nul
if not %ERRORLEVEL%==0 (
  echo EVAL_DB_MISMATCH %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_34 %DATE% %TIME% >> "%LOG%"
  exit /b 34
)
if exist "result\RWKV-fixc.jsonl" del /q "result\RWKV-fixc.jsonl"
if exist "result\RWKV-P-fixc.jsonl" del /q "result\RWKV-P-fixc.jsonl"
del /q scratchpad\eval_shards\shard_*.log 2>nul
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config %DIR%\eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  echo DONE_EXIT_30 %DATE% %TIME% >> "%LOG%"
  exit /b 30
)
echo PHASE 4 OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
endlocal & exit /b 0
