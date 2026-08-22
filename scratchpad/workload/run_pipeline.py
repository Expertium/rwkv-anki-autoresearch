"""Drive the whole replay for a list of users, with a small worker pool.

Per user: canonical table -> FSRS-7 arm (each variant) -> RWKV arm. The RWKV arm is ~99%
of the wall clock (20 reviews/s vs seconds for everything else), so the pool size is
effectively the number of CPU threads this job uses. Each subprocess is pinned to one
thread; the pool size IS the thread budget.

RESUMABLE BY ARTIFACT, not by exit code. A user is skipped only when its output parquet
exists AND its .meta.json parses -- an interrupted arm leaves a parquet-less or
meta-less user that is simply redone. (CLAUDE.md: gate a phase on the ARTIFACT, because a
Python entry point can swallow a fatal error and still exit 0.)

Usage:
  .venv/Scripts/python.exe scratchpad/workload/run_pipeline.py <users.json> [n_workers]
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY_SRSB = Path(r"C:\Users\Andrew\srs-benchmark\.venv\Scripts\python.exe")
WL = ROOT / "scratchpad" / "workload"
TABLES = WL / "tables"
OUT = WL / "out"
FSRS_VARIANTS = ["plain", "sched"]


def done(path):
    meta = Path(str(path) + ".meta.json")
    if not (Path(path).exists() and meta.exists()):
        return False
    try:
        json.loads(meta.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def sh(cmd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ %s\n" % " ".join(str(c) for c in cmd))
        fh.flush()
        rc = subprocess.call([str(c) for c in cmd], cwd=str(ROOT), stdout=fh,
                             stderr=subprocess.STDOUT)
        fh.write("[rc=%d]\n" % rc)
    return rc


def one_user(uid):
    log = WL / "logs" / ("u%d.log" % uid)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table = TABLES / ("u%d.parquet" % uid)
    if not table.exists():
        rc = sh([PY, WL / "build_table.py", uid, table], log)
        if rc != 0:
            return uid, "TABLE_FAIL", 0.0
    for v in FSRS_VARIANTS:
        dest = OUT / ("fsrs%s_u%d.parquet" % ("" if v == "plain" else "_" + v, uid))
        if not done(dest):
            rc = sh([PY_SRSB, WL / "fsrs_arm.py", table, uid, dest, v], log)
            if rc != 0:
                return uid, "FSRS_%s_FAIL" % v, time.time() - t0
    dest = OUT / ("rwkv_u%d.parquet" % uid)
    if not done(dest):
        rc = sh([PY, WL / "rwkv_arm.py", table, uid, dest], log)
        if rc != 0:
            return uid, "RWKV_FAIL", time.time() - t0
    return uid, "OK", time.time() - t0


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    users = spec["users"]
    rows = {int(k): v for k, v in spec["reviews"].items()}
    # biggest first: with a small pool this keeps the long pole off the critical path
    users = sorted(users, key=lambda u: -rows[u])
    print("phase %s: %d users, %d reviews, %d workers"
          % (spec["phase"], len(users), spec["total_reviews"], workers), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for uid, status, dt in ex.map(one_user, users):
            print("  u%-5d %-14s %6.1f min   (elapsed %.2f h)"
                  % (uid, status, dt / 60, (time.time() - t0) / 3600), flush=True)
    print("ALL DONE in %.2f h" % ((time.time() - t0) / 3600), flush=True)


if __name__ == "__main__":
    main()
