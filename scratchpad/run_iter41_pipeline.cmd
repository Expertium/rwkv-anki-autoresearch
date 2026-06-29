@echo off
REM iter41 = moderate deck/preset grow [1,8,3,6,3]. Self-contained DETACHED pipeline (survives Esc/
REM compaction): WS(non-QAT,fast kernel) -> warm-started decay-QAT(card int2/note int4) -> export -> gate.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set LOG=C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\iter41_pipeline.log
set OMP_NUM_THREADS=7
set PYTHONUNBUFFERED=1
echo === PHASE1 WS non-QAT START %DATE% %TIME% === > "%LOG%"
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter41_ws.toml >> "%LOG%" 2>&1
echo === PHASE1 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
copy /Y pretrain\rwkv\opt_iter41\rwkv_iter41_optim_558.pth pretrain\rwkv\opt_iter41\rwkv_iter41_558_optim.pth >> "%LOG%" 2>&1
echo === PHASE2 decay-QAT START %TIME% === >> "%LOG%"
set RWKV_NO_JIT=1
set RWKV_QAT_SCOPE=card:int2,note:int4
.venv\Scripts\python.exe -u -m rwkv.train_rwkv --config rwkv/train_rwkv_config_iter41_qat_decay.toml >> "%LOG%" 2>&1
echo === PHASE2 DONE exit %ERRORLEVEL% %TIME% === >> "%LOG%"
set RWKV_NO_JIT=
set RWKV_QAT_SCOPE=
echo === PHASE3 export %TIME% === >> "%LOG%"
.venv\Scripts\python.exe scratchpad\pth_to_sft.py pretrain\rwkv\opt_iter41\rwkv_iter41_124.pth reference\rwkv_iter41_124.safetensors >> "%LOG%" 2>&1
echo === PHASE4 gate (card int2 + note int4 vs champ fp32) %TIME% === >> "%LOG%"
"C:\Program Files\Git\bin\bash.exe" scratchpad/run_qat_eval.sh reference/rwkv_iter41_124.safetensors >> "%LOG%" 2>&1
echo DONE_EXIT_%ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
