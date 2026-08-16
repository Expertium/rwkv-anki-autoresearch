@echo off
REM Deck-tree smoke: off vs null-map vs real, one subprocess each (ScriptModule bakes env flags).
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\Andrew\rwkv-anki-autoresearch
set OMP_NUM_THREADS=7
set RWKV_ARCH_MODULE=scratchpad/track2_a18/architecture_d80_lora4_cnd.py
set RWKV_INTERLEAVE=1
set RWKV_GRU_HEAD=3
set RWKV_STRIP_L0_VLORA=1
set RWKV_ZERO_FEATURES=22
set RWKV_STATE_CLAMP_TAU=300
set RWKV_STATE_CLAMP_WINDOW=32768
set RWKV_NO_AHEAD_RESIDUAL=1
set RWKV_STRIP_CMIX=user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1
set USERS=%1
if "%USERS%"=="" set USERS=101
set RWKV_DECK_TREE=
set RWKV_DECK_TREE_MAP=
.venv\Scripts\python.exe scratchpad/deck_tree/smoke_tree.py off %USERS%
set RWKV_DECK_TREE=3
set RWKV_DECK_TREE_MAP=scratchpad/deck_tree/parent_maps_null.parquet
.venv\Scripts\python.exe scratchpad/deck_tree/smoke_tree.py null %USERS%
set RWKV_DECK_TREE_MAP=
.venv\Scripts\python.exe scratchpad/deck_tree/smoke_tree.py real %USERS%
echo DONE_EXIT_%ERRORLEVEL%
