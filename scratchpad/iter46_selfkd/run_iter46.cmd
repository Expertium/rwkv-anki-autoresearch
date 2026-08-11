@echo off
REM ===========================================================================================
REM ITER 46: PRIVILEGED SELF-DISTILLATION, imm -> ahead (2026-08-11).
REM Ranked #2/#3 of the 15-proposal set; promoted to next-up because it is the only one of the
REM two gap-exploiting ideas that adds NO deploy debt (see THREE-WAY PARITY below).
REM
REM WHAT IT DOES. RWKV_SELFKD_BETA softens the ahead objective's target away from the raw 0/1
REM label toward the model's OWN better-informed estimate of the same event:
REM     label_y <- b * (1 - P(Again))@teacher_row .detach() + (1-b) * label_y   [BEFORE the KD mix]
REM The teacher row is the QUERY row that scores the very review this row's ahead label refers to
REM (join: review_th[q] == label_review_th[r]). Measured on real LMDB rows: 100.00% coverage of
REM ahead-scored rows, 388,156/388,156 across 5 users, 0 violations
REM (scratchpad/iter46_selfkd/verify_real_data.py).
REM
REM WHY IT SHOULD WORK. research_5k_notes.md "the ahead-vs-imm information gap": identical
REM per-user `size` on all 2500 VAL users, imm better than ahead on 2497 of them, mean gap
REM 0.032411 -- ~100x a typical accepted iteration gain. The model already emits a strictly
REM better-informed estimate of the label the curve head is fit to; this hands it back as a
REM target. Same mechanism as the external-teacher KD that this family has gone 3/3 on
REM (iters 32/35/39, alpha monotone up to 0.9) -- but the teacher is free, online, and aligned.
REM ⚠ The gap is an UPPER BOUND, not a target: distillation transfers the variance-reduction part
REM (a calibrated target beats a 0/1 draw), never the information part -- the query row sees the
REM intervening reviews and the exact lag, and predicting cold from history IS the task.
REM
REM BETA=0.7, and the composition makes this a curriculum rather than a flat dose. Self-KD runs
REM BEFORE the external KD mix and softens only the HARD share, so the target is
REM     a*d128_teacher + (1-a) * [ b*imm_teacher + (1-b)*hard ]
REM with alpha keeping its tuned value EXACTLY (verified by autograd: recovered alpha =
REM 0.899999976 at every beta). On the iter-45 base that is 7% of the WS target (a=0.9) and 35%
REM of the decay target (a=0.5) -- self-distillation bites LATE, when the model's own imm head
REM is actually a good teacher, which is the intended curriculum. That ordering is deliberate;
REM softening the POST-KD target instead would drag alpha 0.9 -> 0.9*(1-b) and bundle two
REM changes into one experiment, which is what iters 42/43/44 were spent un-bundling.
REM
REM THREE-WAY PARITY (CLAUDE.md sec.9 standing directive) -- train / eval / CPU inference:
REM   train : ahead objective, target softened as above.
REM   eval  : UNCHANGED. The loss is not scored; curve_probs (rectified) and p_imm are.
REM   deploy: UNCHANGED. No forward-pass edit, no new parameter (558,212 exactly, asserted in the
REM           smoke test). rust/rwkv-infer needs NOTHING -- unlike the ranked-#2 variant that
REM           feeds R(t) into the Again logit, which would have added a 9th port gap.
REM
REM ⚠ BETA's EFFECTIVE DOSE DEPENDS ON THE BASE -- work it out before changing KDDECAY. The imm
REM teacher's share of the ahead target is (1-alpha)*beta, so at beta=0.5:
REM   KDDECAY=0 (champion base): WS alpha 0.9 -> 5% imm ; decay alpha 0 -> 50% imm ; ~27% average.
REM   KDDECAY=1 (iter-45 base) : WS alpha 0.9 -> 5% imm ; decay alpha 0.5 -> 25% imm ; ~15% average.
REM So if iter 45 ACCEPTS, this same beta delivers roughly HALF the dose. Either raise beta to ~0.7
REM to hold the dose roughly constant, or keep 0.5 and read the result as the weaker point of the
REM lever -- but do not switch the base and silently assume the dose is unchanged.
REM
REM BASE RECIPE -- SET THIS AFTER ITER 45's VERDICT:
REM   KDDECAY=0 -> iter-41 champion base (KD cleared before decay). Gate vs RWKV-iter41_ilv-s0.
REM   KDDECAY=1 -> iter-45 base (KD kept through decay at alpha 0.5). Gate vs RWKV-iter45_kddecay.
REM   ACTIVE: KDDECAY=1, beta 0.7 -> 7% of the WS target, 35% of the decay target.
REM Everything else is the champion env, unchanged.
REM
REM ⚠ Do NOT git rebase/pull/checkout this path while it runs (iter 43's chain died that way).
REM ⚠ NOR edit rwkv/*.py mid-chain: the DECAY phase is a NEW process that imports whatever is on
REM   disk THEN, not what was there at launch. (Found live during iter 45 -- which is why the
REM   prepare_batch hook is gated on the flag and does nothing at all when it is unset.)
REM ⚠ NO del of result jsonls (fresh tags; retries resume from banked users).
REM ===========================================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set DIR=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter46_selfkd
set LOG=%DIR%\iter46.log
set STAMP=%RANDOM%%RANDOM%
set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k
set WSSTEPS=10935
set BETA=0.7
REM ---- BASE: 0 = iter-41 champion, 1 = iter-45 (KD through decay) ----
REM SET TO 1 on 2026-08-11: iter 45 ACCEPTED, so the champion now keeps KD through decay
REM at alpha 0.5. BETA raised 0.5 -> 0.7 with it, because the imm teacher's share is
REM (1-alpha)*beta: on this base the average dose is 0.3*beta vs 0.55*beta on the old one,
REM so an unchanged beta would have delivered barely half the intervention.
set KDDECAY=1

echo ===== ITER 46 (self-distillation imm-^>ahead, beta=%BETA%, kddecay=%KDDECAY%) START %DATE% %TIME% ===== > "%LOG%"

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
REM ---- THE LEVER ----
set RWKV_SELFKD_BETA=%BETA%

echo === WS %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter46_selfkd/i46_ws.toml > "%DIR%\ws_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WSFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 21
)
REM THE experiment's own guard: without it a typo'd flag silently reduces this to the champion
REM and the null would be recorded as a real verdict.
findstr /C:"[selfkd] privileged self-distillation ON" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOSELFKD_WS %DATE% %TIME% >> "%LOG%"
  exit /b 37
)
findstr /C:"hard-share = %BETA% *" "%DIR%\ws_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGBETA_WS %DATE% %TIME% >> "%LOG%"
  exit /b 38
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

REM ---- decay: the base recipe decides whether the external teacher stays on ----
set RWKV_GRAD_STATS=
if "%KDDECAY%"=="1" (
  set RWKV_KD_ALPHA=0.5
) else (
  set RWKV_KD_MIX=
  set RWKV_KD_ALPHA=
)

echo === DECAY SETUP %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_decay_setup.py scratchpad/iter46_selfkd i46_ws i46_d scratchpad/iter46_selfkd/i46_decay.toml train_db_5k_h1 1 5000 1.0 1e-3 65536 > "%DIR%\dsetup_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DSETUPFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 22
)
REM write_decay_setup takes the LATEST ckpt -- assert it took the FINAL WS one, not a mid-run
REM checkpoint from a crashed phase (the guard that would have caught iter 43's chain break).
findstr /C:"i46_ws_%WSSTEPS%" "%DIR%\dsetup_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_WRONGCKPT %DATE% %TIME% >> "%LOG%"
  exit /b 32
)
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config scratchpad/iter46_selfkd/i46_decay.toml > "%DIR%\decay_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_DECAYFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 23
)
REM Self-KD must be ON in DECAY too -- that is where it acts at full strength (a=0 there on the
REM champion base), so losing it in this phase would gut the experiment while WS still logged it.
findstr /C:"[selfkd] privileged self-distillation ON" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOSELFKD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 39
)
REM The BASE recipe's own decay-KD must also be live (iter 45 is the champion now). Losing it would
REM silently compare against a different baseline than the one we are gating vs.
REM ⚠ NOT nested in an if-block: %ERRORLEVEL% inside parentheses expands at PARSE time,
REM so a nested test reads a stale value and the guard silently never fires. goto keeps both
REM the condition and the errorlevel test at top level, where line-by-line parsing makes them
REM correct (the same reason every other guard in this file is written flat).
if not "%KDDECAY%"=="1" goto :skip_basekd
findstr /C:"alpha FIXED at 0.5" "%DIR%\decay_%STAMP%.log" >nul
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_NOBASEKD_DECAY %DATE% %TIME% >> "%LOG%"
  exit /b 40
)
:skip_basekd
echo DECAY OK (selfkd ON) %TIME% >> "%LOG%"

echo === EVAL TOML rect %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad/write_eval_toml.py scratchpad/iter46_selfkd i46_d scratchpad/iter46_selfkd/i46_eval.toml RWKV-iter46_selfkd RWKV-P-iter46_selfkd 5001 7500 > "%DIR%\etoml_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_TOMLFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 24
)
set RWKV_MUON_BATCHED=
set RWKV_NO_JIT=
set RWKV_QAT_COMPILE=
set RWKV_KD_MIX=
set RWKV_KD_ALPHA=
REM Clear the lever for eval: it is a TRAINING-ONLY term, and clearing it also stops prepare()
REM building the teacher index during the (probe-density-1.0) rectified eval.
set RWKV_SELFKD_BETA=
set RWKV_EVAL_PAVA=1
echo === EVAL 5001-7500 RECTIFIED attempt 1 %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter46_selfkd/i46_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval1_%STAMP%.log" 2>&1
if not %ERRORLEVEL%==0 (
  echo EVAL rect attempt 1 failed - retrying from banked users %TIME% >> "%LOG%"
  .venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/iter46_selfkd/i46_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 2 --threads-per-shard 7 > "%DIR%\eval2_%STAMP%.log" 2>&1
)
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 25
)
echo EVAL rect OK %TIME% >> "%LOG%"
endlocal

echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
