"""Parallel feature export across users: split a user list into `--procs` round-robin chunks and run
that many subprocess instances of export_features_fast.py concurrently. Each subprocess is resumable
(skips existing trace_user_*.safetensors), so this is safe to re-run / interrupt. Users are independent
-> ~min(procs, cores) x wall-clock. label_filter_db / parquet reads are concurrent-safe (lmdb readers).

Usage:
  python scratchpad/export_mp.py --procs 8 --range START END
  python scratchpad/export_mp.py --procs 8 U1 U2 U3 ...
Honors RWKV_TRACE_OUT (output dir). Sets RWKV_TORCH_THREADS + OMP_NUM_THREADS per worker so the total
stays ~<= 14 threads (leaves headroom; export is not torch-compute bound anyway).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SCRIPT = str(ROOT / "scratchpad" / "export_features_fast.py")

args = sys.argv[1:]
procs = 8
if "--procs" in args:
    i = args.index("--procs")
    procs = int(args[i + 1])
    del args[i:i + 2]
if args and args[0] == "--range":
    users = list(range(int(args[1]), int(args[2])))
else:
    users = [int(x) for x in args]

# round-robin chunking balances load (later/large users spread across workers)
chunks = [users[i::procs] for i in range(procs)]
chunks = [c for c in chunks if c]
threads = max(1, 14 // len(chunks))

env = dict(os.environ)
env["RWKV_TORCH_THREADS"] = str(threads)
env["OMP_NUM_THREADS"] = str(threads)
env["PYTHONPATH"] = str(ROOT)
out = env.get("RWKV_TRACE_OUT", "reference")

print(f"exporting {len(users)} users across {len(chunks)} procs ({threads} threads/proc) -> {out}",
      flush=True)
t0 = time.time()
running = []
for c in chunks:
    cmd = [PY, SCRIPT] + [str(u) for u in c]
    running.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
rcs = [p.wait() for p in running]
dt = time.time() - t0
print(f"DONE {len(users)} users in {dt:.1f}s "
      f"({len(users) / dt:.2f} users/s); procs={len(chunks)} threads/proc={threads} rcs={rcs}",
      flush=True)
