"""One-shot: append the 2026-11-21 22:55 board amendment (deadlock fix + corruption chain).

Appends with LF, matching the newest entries the hourly supervisor writes.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[3] / "optimization" / "GPU_RUN_BOARD.md"
BLOCK = """
Row 11 amendment 14 (2026-11-21 22:55, Moscow). **The three-chain deadlock is fixed and the queue is
moving again. The corruption/finiteness check is queued behind it.** Fix (`_fix_chain_busypat2.py`,
22:20): `BUSY_PAT` in `it36rankprec`, `it36longdecay` and `it36shrink` listed each other's chain
SCRIPT, so each saw its two siblings' waiting processes forever (`matching processes=4`) and none
could take the GPU; all three now match real GPU work only
(`qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|rwkv-infer`).
The waiters were killed, the three stale locks deleted, and the chains relaunched 22:21:59-22:22:00.
`it36rankprec` ran the `l2i4s4` 0.04-epoch smoke, its 124-step QAT decay (6.1 min, rc=0) and started
the quantized eval at 22:29:18; `it36longdecay` and `it36shrink` wait behind it correctly. Cost of the
deadlock: the GPU was idle 12:25-22:23, about 10 h.

New chain `it36corrupt` (launched 22:50, waits for `it36shrink`): the corruption check Andrew asked for
at 23:52 on 2026-11-20 - score the 50 users with the longest average review histories, with their
histories NOT truncated, and fail if even ONE card/note state value or even ONE p(recall) is
non-finite. The eval's own guards cannot answer it: `_get_loss` returns None only on NaN (never on
+inf) and the QAT state path deliberately caps non-finite states before quantizing
(`_sanitize_state`, `rwkv_ops.py:313`), which is exactly what hides a blow-up. The probe
(`scratchpad/quant_plan/corruption/finite_probe.py`) therefore monkey-patches the eval path - no repo
source is edited - and watches the RAW state, the shift vectors and every p_curve/p_imm/p_imm_all at
`RWKV_QAT_FUSED=0` (the documented bit-parity Python twin of the fused kernels), one process, no
sharding, on `corruption/corrupt50.json` (50 users, 4,609,289 section rows, longest single-card
history 1,409; the 62 users whose full section exceeds the 200,000-row VRAM cap are recorded in that
file's `excluded_too_big` and backfilled - Andrew: "skip users that cannot fit into VRAM"). Points, in
value order: `l2i8s4` (row 17), `l2i8` (15), `l3i8` (16), `l3i8s4` (18), then the quantization-OFF
control (`qat_it36ctl/ctl_124.pth`) for attribution. CPU smoke (user 7325, full history, `l2i8`): 828
state calls / 189,923,328 state values, maxabs 0.345, 0 non-finite, 3 p(recall) tensors clean -> PASS.
Two guards were added with it: `status_report.py`'s WORKLOAD pattern now includes
`probe_get_result|check_corruption` (the check runs `get_result` under its own entry point, so without
this the report would have called the GPU idle while it scored), and the chain logs a heartbeat every
5 minutes while a point runs (a 50-user point can take an hour, and a silent `run()` would have looked
like a dead chain to the supervisor).
"""
with P.open("a", encoding="utf-8", newline="") as fh:
    fh.write(BLOCK)
print(f"appended {len(BLOCK)} chars to {P}")
