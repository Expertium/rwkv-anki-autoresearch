# AUTORUN LOG - the hourly quant supervisor

One block per run, newest at the bottom. The supervisor obeys `QUEUE.md` only.

## 2026-11-21 09:38 autorun
- state: GPU BUSY - `it36rank` chain training the `l3i8s4` row-18 QAT decay (qat_decay.py + train_rwkv, 4 workload processes, GPU 1,896 MiB / 19 %); four other LIVE chains waiting on their predecessor locks; host RAM free 40.2 GiB; RAM guard heartbeat 0.1 min old
- did: section 4 only - recorded the landed sweep `it36l2i8s4` (l2i8s4: card rank-2 int8 + int4 shifts, note rank-1 int8 + int4 shifts; 0.333853/0.306539 = +0.000052 ub95 +0.000118 ahead, +0.000327 ub95 +0.000390 imm, PASS): filled RESULTS.md row 17 (was "running"), updated its `**Status:**` line to rows 15-17, added the stamp `it36l2i8s4` to `autorun/RECORDED.txt` (and replaced that file's now-stale "deliberately ABSENT" note), added one bullet at the top of NOTES.md `## Live status`. No launch, no restart, no kill, no file moves.
- next: the GPU should still be busy with `l3i8s4` (row 18) at the next run, then `it36notecut` / `it36rankprec` / `it36longdecay` / `it36shrink` take the lock in turn; the next run should record row 18 when `quant_sweep_it36l3i8s4.json` lands. The idle deploy measurement (`rust_speed_idle_l2i8.json`) is still owed and stays blocked while any chain holds the GPU.
- verdict: OK

## 2026-11-21 09:41 autorun
- state: GPU BUSY - `it36rank` chain evaluating the `l3i8s4` row-18 sweep (quant_sweep.py + eval_sharded.py, `--shards 2`, 4 workload processes, pid 25928 started 09:38:33); GPU 4,853 MiB / 95 %; four other LIVE chains waiting on their predecessor locks; host RAM free 37.2 GiB; RAM guard heartbeat 0.1 min old
- did: nothing. Step 2.1 (GPU BUSY) forbids launch, kill, restart and modification; QUEUE.md section 4 had 0 NOT RECORDED results (`quant_sweep_it36l3i8s4.json` has not landed yet, it is the sweep running now), so there was nothing to record. No file edits except this log block.
- next: record RESULTS.md row 18 when `quant_sweep_it36l3i8s4.json` lands (stamp `it36l3i8s4`), then `it36notecut` / `it36rankprec` / `it36longdecay` / `it36shrink` take the lock in turn. The idle deploy measurement (`rust_speed_idle_l2i8.json`) is still owed and stays blocked while any chain holds the GPU.
- note for Andrew: `status_report.py` prints "QUEUE.md pending launch items: 1" because its pattern also matches the literal format example at QUEUE.md line 79 (`- [ ] <what to run> :: ...`). Section 5 itself says "None right now", so this is a parser false positive, not an authorization - I did not act on it. Rewording that example line (for example prefixing it with `#` or writing it as `<what to run> :: <cmd> :: <proof>` without the box) would clear it. I left QUEUE.md unchanged.
- verdict: OK