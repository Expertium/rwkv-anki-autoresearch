@echo off
REM iter43 = NOTE INT2 target test. Decay-QAT from the iter36 CHAMPION WS-final (warm-start), card int2
REM + NOTE INT2. Self-contained DETACHED pipeline (survives Esc/compaction). Restores champion arch first
REM (current architecture.py is the rejected iter42 grow), then: decay-QAT -> export -> gate(card int2/note int2).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter43_pipeline.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === ARCH RESTORE -^> arch_iter36.py [1,4,3,3,3] champion %DATE% %TIME% === > "%LOG%"
copy /Y optimization\arch_snapshots\arch_iter36.py rwkv\architecture.py >> "%LOG%" 2>&1
if not exist pretrain\rwkv\opt_qat43 mkdir pretrain\rwkv\opt_qat43
echo === PHASE1 decay-QAT (card int2, note int2) START %TIME% === >> "%LOG%"
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int2
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter43_qat_decay.toml >> "%LOG%" 2>&1
echo === PHASE1 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
set RWKV_NO_JIT=
set RWKV_QAT_SCOPE=
echo === PHASE2 export %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\pth_to_sft.py pretrain\rwkv\opt_qat43\rwkv_iter43_124.pth reference\rwkv_iter43_124.safetensors >> "%LOG%" 2>&1
echo === PHASE3 gate (card int2 + note int2 vs champ fp32) %TIME% === >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/run_qat_eval.sh reference/rwkv_iter43_124.safetensors card:int2,note:int2 >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
