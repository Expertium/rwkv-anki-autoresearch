@echo off
REM ===========================================================================================
REM ITER 50: THE DECK TREE (RWKV_DECK_TREE=3). Family: topology. Andrew's own long-standing ask:
REM   card-note-deck-preset-global  becomes  card-note-(deck, depth_level)-preset-global
REM
REM THE LEVER: the deck stream is applied once per DEPTH LEVEL of the user's deck tree, grouping
REM reviews by the deck's k-th ANCESTOR. Chain becomes
REM   card_id, note_id, deck_id, deck_id@1, deck_id@2, preset_id, user_id
REM The SAME deck module object runs at every level (weight sharing), so depth is a LOOP COUNT
REM over the user's tree, not an architecture constant. The only new weights are a level
REM embedding, 2 x 80 == 160 floats, added at each level's layer 0 so the shared module knows
REM which scope it is running at.
REM
REM WHY LEVEL 3 AND NOT 2: the deck-depth histogram PEAKS AT 4, not 1 (level_reach.py, 40 users,
REM 3.67 M reviews, review-weighted). Reach by ancestor distance: 1 == 49.21 pct, 2 == 38.29 pct,
REM 3 == 31.20 pct, 4 == 20.93 pct. A single parent link would test "parent deck", not "tree".
REM L=3 is the smallest L that actually tests the hypothesis Andrew stated.
REM
REM NO LMDB REBUILD: data_processing drops parent_id but never factorizes deck_id, so the
REM parquet's deck_id-to-parent_id map applies directly to ids already in the LMDB. Confirmed at
REM scale in optimization/FUTURE_FEATURES.md.
REM
REM ROWS WITH NO k-TH ANCESTOR bypass EXACTLY: they get a row-unique negative id (a singleton
REM sequence, the cheapest thing the kernel can be handed) and are marked inactive, and the model
REM never scatters an inactive row back, so x keeps its incoming value. Proven, not asserted: an
REM all-inactive parent map reproduces the tree-off forward BIT-FOR-BIT in both Python paths
REM (scratchpad/deck_tree/smoke_tree.py, scratchpad/parity3/smoke_deck_tree_rnn.py).
REM
REM SINGLE VARIABLE vs the iter-45 champion: this file is run_iter45.cmd with the prefixes
REM changed and TWO lines added (RWKV_DECK_TREE=3 and the param assert). Seed 4321, KD alpha 0.9
REM WS / 0.5 decay, PAVA 0.2, tuned HPs, the speed stack: all identical.
REM
REM KNOWN CONFOUND, stated up front: 13 layer-steps become 21, so this run has about 1.6x the
REM model compute of the champion. A win is therefore "the tree helps" OR "more deck compute
REM helps", and the disambiguating follow-up is the L=2 arm (17 layer-steps) plus a depth-matched
REM control. A TIE needs no disambiguation, which is the usual asymmetry.
REM
REM DEPLOY: srs_model_rnn.py already mirrors it (ancestor states are handed back through the
REM caller's list). Rust port only if this is ACCEPTED; it would add per-deck ancestor states,
REM which the gate explicitly allows to grow (card and note state are untouched).
REM
REM Do NOT git checkout / edit this file while it runs (iters 43 and 46 died that way).
REM NO del of result jsonls (fresh tag; retries resume from banked users).
REM No angle brackets or arrows in REM lines -- cmd parses redirection before REM.
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter50_decktree
set LOG=%DIR%\iter50.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935

echo ===== ITER 50 (deck tree, RWKV_DECK_TREE=3) START %DATE% %TIME% ===== > "%LOG%"

setlocal
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=4321
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_PAVA_LAMBDA=0.2
set RWKV_PROBE_DENSITY=0.08
set RWKV_PROBE_DUR=0.0
set RWKV_MUON=1
set RWKV_MUON_LR=0.0025
set RWKV_MUON_MOMENTUM=0.95
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set RWKV_WEIGHT_DECAY=0.01
set RWKV_WEIGHT_DECAY_HEAD=0.01
set RWKV_CLIP=0.25
set RWKV_ADAMW_BETA2=0.999
set RWKV_DROPOUT_SCALE=0.5
set RWKV_GRAD_STATS=%DIR%\grad_stats.json
set RWKV_MUON_BATCHED=1
set RWKV_NO_JIT=1
set RWKV_QAT_COMPILE=1
set RWKV_KD_MIX=%DUMP%:%WSSTEPS%
set RWKV_KD_ALPHA=0.9

REM ================= THE LEVER, and the ONLY behavioural difference from run_iter45.cmd ========
set RWKV_DECK_TREE=3

REM PHASE 0 GUARD -- the param count is CONSUMED state, unlike a banner (the QAT-inert bug printed
REM a truthful banner for an object discarded one line later). Champion 558,212; the deck tree
REM adds EXACTLY the 160-float level embedding because the deck module is shared. If the module
REM were accidentally duplicated instead of shared, this lands at 584k+ and stops here, before any
REM GPU is spent. It also catches the config-sharing bug found while building this: one shared
REM config object let the last level overwrite stream_name, which silently disabled
REM RWKV_STRIP_CMIX's deck_id entries and added 26,070 params with no error anywhere.
.venv\Scripts\python.exe scratchpad\parity3\assert_param_count.py 558372 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_PARAMMISMATCH %DATE% %TIME% >> "%LOG%"
  exit /b 44
)

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter50_decktree/i50_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
findstr /C:"[deck-tree] ON: 3 deck levels" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOTREE_WS %DATE% %TIME% >> "%LOG%"
  exit /b 39
)
findstr /C:"[interleave] round-robin layer schedule ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOILV %DATE% %TIME% >> "%LOG%"
  exit /b 28
)
findstr /C:"placement = front-loaded" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGPLACEMENT %DATE% %TIME% >> "%LOG%"
  exit /b 33
)
echo WS OK %TIME% >> "%LOG%"

set RWKV_KD_ALPHA=0.5
set RWKV_GRAD_STATS=

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter50_decktree i50_ws i50_d scratchpad/iter50_decktree/i50_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter50_decktree/i50_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
findstr /C:"[deck-tree] ON: 3 deck levels" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOTREE_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 40
)
findstr /C:"[kd-mix] KD ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOKD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 35
)
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGALPHA_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 36
)
echo DECAY OK (TREE ON, KD ON, alpha 0.5) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter50_decktree i50_d scratchpad/iter50_decktree/i50_eval.toml RWKV-iter50_decktree RWKV-P-iter50_decktree 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
REM RWKV_DECK_TREE stays SET for the eval -- it is a MODEL property, not a training trick, and the
REM checkpoint carries tree_level_emb. The speed stack IS cleared, as always. NO_JIT is cleared, so
REM the eval SCRIPTS the model: verified on CPU under this exact arch env that scripting compiles
REM AND that the scripted forward_batch RUNS (iter 48 died on a scripted-only runtime bug that a
REM compile-only check had passed).
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter50_decktree/i50_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter50_decktree/i50_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
exit /b 0
