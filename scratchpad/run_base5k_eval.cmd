@echo off
REM Task #8: d=128 baseline eval on 5001-10000 (the 5k-phase target number). Swaps in the OLD
REM architecture, runs get_result, ALWAYS restores the champion architecture, then computes the
REM by-user mean LogLoss for both modes. Log: scratchpad/base5k_eval.log (end marker EVAL5K_EXIT_n).
REM Safe under LMDB-build CPU contention: logloss is load-independent (only wall time inflates).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PY=.venv\Scripts\python.exe
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\base5k_eval.log
set PYTHONUNBUFFERED=1
REM the OLD arch defines its own heads/dims -- clear champion env overrides; NO QAT (target stays fp)
set RWKV_N_HEADS=
set RWKV_HEAD_DIM=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_PQ=
set RWKV_QAT_FUSED=
set RWKV_QAT_SHIFT_SCOPE=
set RWKV_DETERMINISTIC=1
set RWKV_AUGMENT_SEED=1234
set RWKV_EMPTY_CACHE_EVERY=0

echo ===== BASE5K EVAL START %DATE% %TIME% ===== >> "%LOG%"
copy /y rwkv\architecture.py scratchpad\architecture_champion_backup.py >> "%LOG%" 2>&1
copy /y scratchpad\architecture_old_d128.py rwkv\architecture.py >> "%LOG%" 2>&1
del /q result\RWKV-base5k.jsonl result\RWKV-P-base5k.jsonl 2>nul

%PY% -u -m rwkv.get_result --config rwkv/get_result_config_base5k.toml >> "%LOG%" 2>&1
set EVAL_EXIT=%ERRORLEVEL%

copy /y scratchpad\architecture_champion_backup.py rwkv\architecture.py >> "%LOG%" 2>&1
echo [ARCH RESTORED] %DATE% %TIME% >> "%LOG%"

%PY% -c "import json; f=lambda p:[json.loads(l)['metrics']['LogLoss'] for l in open(p)]; a=f('result/RWKV-base5k.jsonl'); i=f('result/RWKV-P-base5k.jsonl'); print(f'BASE5K_RESULT ahead={sum(a)/len(a):.6f} (n={len(a)}) imm={sum(i)/len(i):.6f} (n={len(i)})')" >> "%LOG%" 2>&1

echo EVAL5K_EXIT_%EVAL_EXIT% %DATE% %TIME% >> "%LOG%"
