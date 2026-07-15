@echo off
REM ============================================================================
REM TRACK 2 A0 -- EVAL RESUME (2026-07-15). The first eval crashed at user 6701:
REM the 1-ep d=128 model NaNs on 1,048,576-token mega-chunks (get_loss NaN guard
REM -> stats None -> AttributeError), and get_result's old except swallowed it to
REM exit 0 with 1700/5000 users merged. Fixes now in: get_result re-raises,
REM NaN-skips + records such users (result/<FILE_AHEAD>.nanskip.jsonl), resumes
REM past both done and skipped users; eval_sharded gates completeness. This .cmd
REM re-runs ONLY eval + the informational paired test. The -s0 shard files (1700
REM users) are the resume basis -- only the bad canon merges are deleted.
REM ============================================================================
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\track2_a0\track2_a0.log
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=1
set RWKV_EMPTY_CACHE_WINDOW=0
set RWKV_ARCH_MODULE=scratchpad/architecture_old_d128.py

echo ===== TRACK2_A0 EVAL-RESUME START %DATE% %TIME% ===== >> "%LOG%"
del /Q result\RWKV-track2_a0.jsonl result\RWKV-P-track2_a0.jsonl 2>nul
echo === EVAL (single process, resume past 1700 done users) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe -u optimization/eval_sharded.py --config scratchpad/track2_a0/track2_a0_eval.toml --shards 1 --solo-threshold 0 --fetch-per-shard 4 --threads-per-shard 7 >> "%LOG%" 2>&1
if not %ERRORLEVEL%==0 (
  echo DONE_EXIT_EVALFAIL_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
  exit /b 7
)

echo === INFO: paired vs upstream 12-ep d=128 (budget check; user-set mismatch expected if NaN-skips) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe optimization/paired_pvalue.py --cand-ahead result/RWKV-track2_a0.jsonl --cand-imm result/RWKV-P-track2_a0.jsonl --champ-ahead result/RWKV-base5k.jsonl --champ-imm result/RWKV-P-base5k.jsonl >> "%LOG%" 2>&1
echo GATE_DONE (informational paired_pvalue exit %ERRORLEVEL%) >> "%LOG%"
echo DONE_EXIT_0 %DATE% %TIME% >> "%LOG%"
