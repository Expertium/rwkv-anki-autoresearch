@echo off
REM iter46 = FIRST REAL low-rank QAT (card rank-2 int4 + note int2), 8-epoch decay from the champion.
REM Self-contained DETACHED pipeline. architecture.py is already champion+QAT-parser; restore defensively
REM (the synced arch_iter36 snapshot now CONTAINS the parser). decay-QAT -> export latest -> honest gate.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter46_pipeline.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === ARCH RESTORE -^> arch_iter36 (champion + QAT parser) %DATE% %TIME% === > "%LOG%"
copy /Y optimization\arch_snapshots\arch_iter36.py rwkv\architecture.py >> "%LOG%" 2>&1
if not exist pretrain\rwkv\opt_qat46 mkdir pretrain\rwkv\opt_qat46
echo === PHASE1 low-rank QAT (card rank2:int4, note int2) START %TIME% === >> "%LOG%"
set RWKV_NO_JIT=1
set RWKV_QAT_LOWRANK_SCOPE=card:2:int4
set RWKV_QAT_SCOPE=note:int2
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter46_qat_decay.toml >> "%LOG%" 2>&1
echo === PHASE1 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
set RWKV_NO_JIT=
set RWKV_QAT_LOWRANK_SCOPE=
set RWKV_QAT_SCOPE=
echo === PHASE2 export (latest checkpoint) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\export_latest.py pretrain\rwkv\opt_qat46 rwkv_iter46 reference\rwkv_iter46.safetensors >> "%LOG%" 2>&1
echo === PHASE3 honest gate (card rank2:int4 + note int2 + shifts) %TIME% === >> "%LOG%"
set RWKV_QUANT_SHIFTS=1
"C:\Program Files\Git\bin\bash.exe" scratchpad/run_qat_eval.sh reference/rwkv_iter46.safetensors "note:int2" 12 "card:2:int4" >> "%LOG%" 2>&1
set RWKV_QUANT_SHIFTS=
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
