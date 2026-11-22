@echo off
REM Export deploy traces for the 50 longest-history users (the corruption gate's roster) so the RUST
REM finiteness check (RWKV_FINITE_CHECK=1) can be pointed at the same worst-case users as the Python
REM probe. Started 2026-11-22. Five shards: the per-review loop is overhead-bound (one RNN step per
REM review, ~28 module forwards each), so one process per shard beats widening the thread pool.
REM Sharding is safe -- export_user() seeds torch per USER, so no trace depends on its process.
REM Per-shard logs below; resume state is trace_corrupt50\export_progress_shard<i>.jsonl.
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
for %%i in (0 1 2 3 4) do (
  start "trace_shard%%i" /b cmd /c ".venv\Scripts\python.exe -X utf8 -u scratchpad\quant_plan\corruption\export_corrupt50_traces.py --shard %%i --nshards 5 --threads 3 > scratchpad\quant_plan\trace_corrupt50\export_shard%%i.log 2>&1"
)
echo launched 5 shards
