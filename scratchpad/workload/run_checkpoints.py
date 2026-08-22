"""Drive the v2 checkpoint arm over every user that already has an RWKV replay.

Only the FSRS side is rebuilt. RWKV's weights are frozen and user-independent, so the
intervals from the v1 run are already what it would have produced at any checkpoint with
only the past in hand -- there is nothing to re-derive.

Resumable by artifact: a user is skipped when its checkpoint parquet AND its meta both
exist and parse.

Usage: .venv/Scripts/python.exe scratchpad/workload/run_checkpoints.py [n_workers] [step_days]
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY_SRSB = Path(r"C:\Users\Andrew\srs-benchmark\.venv\Scripts\python.exe")
WL = ROOT / "scratchpad" / "workload"
CP = WL / "cp"


def done(p):
    m = Path(str(p) + ".meta.json")
    if not (Path(p).exists() and m.exists()):
        return False
    try:
        json.loads(m.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def one(args):
    uid, step = args
    dest = CP / ("cp_u%d.parquet" % uid)
    if done(dest):
        return uid, "SKIP", 0.0
    log = WL / "logs" / ("cp_u%d.log" % uid)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(
            [str(PY_SRSB), str(WL / "checkpoint_arm.py"),
             str(WL / "tables" / ("u%d.parquet" % uid)), str(uid), str(dest), str(step)],
            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    return uid, ("OK" if rc == 0 and done(dest) else "FAIL_%d" % rc), time.time() - t0


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    # every user with a finished RWKV replay AND a canonical table
    users = sorted(int(p.stem.split("_u")[1]) for p in (WL / "out").glob("rwkv_u*.parquet")
                   if (WL / "tables" / ("u%s.parquet" % p.stem.split("_u")[1])).exists())
    # biggest first keeps the long pole off the critical path with a small pool
    import pandas as pd
    size = {u: len(pd.read_parquet(WL / "tables" / ("u%d.parquet" % u), columns=["day_offset"]))
            for u in users}
    users.sort(key=lambda u: -size[u])
    print("checkpoint arm: %d users, step %d days, %d workers"
          % (len(users), step, workers), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for uid, status, dt in ex.map(one, [(u, step) for u in users]):
            print("  u%-6d %-8s %6.1f min   (elapsed %.2f h)"
                  % (uid, status, dt / 60, (time.time() - t0) / 3600), flush=True)
    print("ALL DONE in %.2f h" % ((time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()
