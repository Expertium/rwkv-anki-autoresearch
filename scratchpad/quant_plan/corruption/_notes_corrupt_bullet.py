"""One-shot: insert the 2026-11-21 22:55 bullet at the top of NOTES.md's `## Live status` list.

NOTES.md is mixed-endings (mostly CRLF), so this edits in text mode with newline='' and keeps whatever
ending the anchor lines already use.
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "NOTES.md"
BULLET = (
    "* 2026-11-21 22:55 - **The 10-hour three-chain deadlock is fixed, and the corruption check is "
    "queued behind the running queue.** (1) Fix: `it36rankprec`, `it36longdecay` and `it36shrink` each "
    "listed the other two chain SCRIPTS in `BUSY_PAT`, so each saw its siblings' waiting processes "
    "forever (`matching processes=4`) and none could take the GPU - idle 12:25-22:23, about 10 h. "
    "`_fix_chain_busypat2.py` rewrote all three to real GPU work only "
    "(`qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
    "rwkv-infer`); the waiters were killed, the stale locks deleted, and the three chains relaunched "
    "22:21:59-22:22:00. `it36rankprec` ran the `l2i4s4` smoke, its 124-step decay (6.1 min, rc=0) and "
    "started the quantized eval at 22:29:18; `it36longdecay` and `it36shrink` wait behind it correctly. "
    "(2) New chain `it36corrupt` (launched 22:50, waits for `it36shrink`): Andrew's corruption gate - "
    "score the 50 users with the longest average review histories, histories NOT truncated, and fail if "
    "even ONE card/note state value or even ONE p(recall) is non-finite. The eval's own guards cannot "
    "answer it (`_get_loss` returns None only on NaN, never on +inf, and the QAT state path caps "
    "non-finite states before quantizing, `_sanitize_state` at `rwkv_ops.py:313`), so "
    "`corruption/finite_probe.py` monkey-patches the eval path (no repo source edited) and watches the "
    "RAW state, the shift vectors and every `p_curve`/`p_imm`/`p_imm_all`, at `RWKV_QAT_FUSED=0` (the "
    "documented bit-parity Python twin of the fused kernels) - one process, no sharding, on "
    "`corruption/corrupt50.json` (50 users, 4,609,289 section rows; the 62 users over the 200,000-row "
    "VRAM cap are recorded in `excluded_too_big` and backfilled). Points in value order: `l2i8s4` "
    "(row 17), `l2i8` (15), `l3i8` (16), `l3i8s4` (18), then the quantization-OFF control "
    "(`qat_it36ctl/ctl_124.pth`). CPU smoke (user 7325, full history, `l2i8`): 828 state calls / "
    "189,923,328 state values, maxabs 0.345, 0 non-finite, 3 p(recall) tensors clean -> PASS. Two "
    "guards came with it: `status_report.py`'s WORKLOAD pattern now includes "
    "`probe_get_result|check_corruption`, and the chain logs a heartbeat every 5 minutes while a point "
    "runs (a point can take an hour, and a silent `run()` would look like a dead chain to the hourly "
    "supervisor). Verdicts land in `corruption/corruption_corrupt_<tag>.json` and become a corruption "
    "column in RESULTS.md.\r\n"
    "\r\n"
)

with P.open(encoding="utf-8", newline="") as fh:
    text = fh.read()
if "22:55 - **The 10-hour three-chain deadlock" in text:
    raise SystemExit("bullet already present")
for nl in ("\n", "\r\n"):
    anchor = f"## Live status{nl}{nl}"
    if anchor in text:
        bullet = BULLET.replace("\r\n", nl)
        text = text.replace(anchor, anchor + bullet, 1)
        break
else:
    raise SystemExit("anchor not found (neither the LF nor the CRLF form of the Live status header)")
with P.open("w", encoding="utf-8", newline="") as fh:
    fh.write(text)
print("inserted")
