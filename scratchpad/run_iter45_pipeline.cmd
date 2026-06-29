@echo off
REM iter45 = QAT plateau probe: LONGER 16-epoch decay-QAT from the iter36 champion, card int2 + note int2.
REM Self-contained DETACHED pipeline (survives Esc/compaction). Restores champion arch first (defensive),
REM then: decay-QAT(16ep) -> export latest checkpoint -> gate(card int2/note int2).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter45_pipeline.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === ARCH RESTORE -^> arch_iter36.py [1,4,3,3,3] champion %DATE% %TIME% === > "%LOG%"
copy /Y optimization\arch_snapshots\arch_iter36.py rwkv\architecture.py >> "%LOG%" 2>&1
if not exist pretrain\rwkv\opt_qat45 mkdir pretrain\rwkv\opt_qat45
echo === PHASE1 decay-QAT 16ep (card int2, note int2) START %TIME% === >> "%LOG%"
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int2
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter45_qat_decay.toml >> "%LOG%" 2>&1
echo === PHASE1 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
set RWKV_NO_JIT=
set RWKV_QAT_SCOPE=
echo === PHASE2 export (latest checkpoint) %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\export_latest.py pretrain\rwkv\opt_qat45 rwkv_iter45 reference\rwkv_iter45.safetensors >> "%LOG%" 2>&1
echo === PHASE3 gate (card int2 + note int2 vs champ fp32) %TIME% === >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/run_qat_eval.sh reference/rwkv_iter45.safetensors card:int2,note:int2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
