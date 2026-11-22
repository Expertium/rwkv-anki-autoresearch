"""Hold the GPU queue for one short, explicit window (self-expiring).

Why: every `it36*_chain.py` waiter decides "the GPU is free" by counting python/rwkv-infer processes
whose COMMAND LINE matches its own BUSY_PAT (`it36ladder_lib.running`). Chains hand off inside a
point with no idle gap, and every waiter polls once a minute, so a non-chain GPU job that waits for a
free GPU races the waiters at each handoff. This script removes the race for a bounded window: while it
runs, its command line carries `--tag` (a token that IS one of those BUSY_PAT alternatives), so the
waiters keep waiting until this process exits.

It is a QUEUE HOLD, not GPU work: the script only sleeps, so it does not use the GPU, and it can never
create a second GPU workload. It also always exits after `--secs` seconds, so a crash in the operator's
flow cannot wedge the queue.

Usage (2026-11-22 corruption sweep for RESULTS rows 29-32):
  .venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\corruption\\gpu_hold.py --tag rust_grid.py --secs 3600
"""

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rust_grid.py",
                    help="a token that appears in the chain waiters' BUSY_PAT (the hold matches on it)")
    ap.add_argument("--secs", type=int, default=3600)
    args = ap.parse_args()

    t0 = time.time()
    print(f"GPU queue hold ACTIVE: pid={__import__('os').getpid()} tag={args.tag} "
          f"secs={args.secs} (chain waiters will read 'busy')", flush=True)
    while time.time() - t0 < args.secs:
        left = int(args.secs - (time.time() - t0))
        print(f"hold: {left} s left", flush=True)
        time.sleep(60)
    print("GPU queue hold RELEASED (time limit reached); waiters may now take the GPU", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
