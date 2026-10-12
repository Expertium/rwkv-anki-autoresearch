"""RAM guard (Andrew 2026-09-15: "I REALLY need to always keep at least 10 GB of RAM free").

WHY. A fetch worker reads the LMDB through a memory map, and Windows keeps every touched page in the
worker's WORKING SET until memory pressure trims it -- so each of the two decay/eval workers grows to
~23 GB of file pages (private memory only ~3 GB) and "available" RAM falls to 0.4-4 GB even though the
pages are reclaimable. Available RAM is what Andrew watches and what his own programs need.

WHAT. Every 5 s: for each python.exe fetch worker of an rwkv run (a `multiprocessing` spawn child whose
parent runs rwkv.train_rwkv / get_result / eval_sharded / eval_rows / probe_buttons), keep a HARD working-set
maximum via SetProcessWorkingSetSizeEx(QUOTA_LIMITS_HARDWS_MAX_ENABLE): Windows then trims the excess
(least-recently-used file pages) to the standby list, which counts as available. The cap is DYNAMIC:
  * a new worker starts at CAP_MIN (6 GB);
  * while available RAM is above TARGET + 2 GB (TARGET = FLOOR + 4 = 14 GB) each worker's cap is set to its
    RSS + its share of the spare (available - TARGET), up to CAP_MAX (16 GB) -- the working set then grows only as the worker touches pages, so this converges
    gently and the VALIDATION pages (re-read every ~2000 steps from the F: HDD at ~3 MB/s = ~7 min cold) stay
    resident between validations. Measured 2026-09-15: a fixed 6 GB cap made every validation cold (the train
    stream is ~0.2 GB/min per worker, so 43 min of it evicts the validation pages); 10-12 GB keeps them;
  * when available falls under FLOOR + 1.5 GB the caps are lowered at once by the deficit (never under CAP_MIN),
    and if available is still under FLOOR after that, EmptyWorkingSet() runs once a minute.
The trainer / eval process itself is never touched: its private memory is real. Numerics are untouched by
construction (memory placement only). Log: scratchpad/ram_guard.log.

Usage: python scratchpad/ram_guard.py            (loop; run detached via run_ram_guard.cmd)
       python scratchpad/ram_guard.py --once     (one pass, prints what it did)
"""
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time

import psutil

CAP_MIN = float(os.environ.get("RWKV_RAM_GUARD_CAP_GB", "6"))        # the cap a worker starts at and never goes under
CAP_MAX = float(os.environ.get("RWKV_RAM_GUARD_CAP_MAX_GB", "16"))   # the most a worker is ever allowed
FLOOR_GB = float(os.environ.get("RWKV_RAM_GUARD_FLOOR_GB", "10"))    # Andrew: always >= 10 GB available
TARGET_GB = FLOOR_GB + 4.0                                            # raise caps only while available > TARGET + 2
POLL_S = 5
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ram_guard.log")
PARENT_MARKS = ("rwkv.train_rwkv", "rwkv.get_result", "eval_sharded", "eval_rows", "probe_buttons", "get_result_cf")

k32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
PROCESS_SET_QUOTA, PROCESS_QUERY_INFORMATION = 0x0100, 0x0400
QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x2, 0x4
k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = (wt.DWORD, wt.BOOL, wt.DWORD)
k32.SetProcessWorkingSetSizeEx.restype = wt.BOOL
k32.SetProcessWorkingSetSizeEx.argtypes = (wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wt.DWORD)
psapi.EmptyWorkingSet.restype = wt.BOOL
psapi.EmptyWorkingSet.argtypes = (wt.HANDLE,)
k32.CloseHandle.argtypes = (wt.HANDLE,)


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="ascii", errors="replace") as f:
        f.write(line + "\n")


def cmdline(p):
    try:
        return " ".join(p.cmdline() or [])
    except Exception:
        return ""


def fetch_workers():
    """python.exe multiprocessing children of an rwkv train/eval process."""
    out = []
    for p in psutil.process_iter(["pid", "name", "ppid"]):
        try:
            if (p.info["name"] or "").lower() != "python.exe":
                continue
            cl = cmdline(p)
            if "multiprocessing" not in cl:
                continue
            par = psutil.Process(p.info["ppid"])
            pcl = cmdline(par)
            if any(m in pcl for m in PARENT_MARKS):
                out.append((p, par))
        except Exception:
            continue
    return out


def cap(pid, gb):
    h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        return False, ctypes.get_last_error()
    ok = k32.SetProcessWorkingSetSizeEx(h, int(0.2 * 2**30), int(gb * 2**30),
                                        QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)
    err = 0 if ok else ctypes.get_last_error()
    k32.CloseHandle(h)
    return bool(ok), err


def empty(pid):
    h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        return False
    ok = psapi.EmptyWorkingSet(h)
    k32.CloseHandle(h)
    return bool(ok)


def avail_gb():
    return psutil.virtual_memory().available / 2**30


def one_pass(caps, last_empty):
    """caps: pid -> current hard cap in GB (None = the cap call failed). Returns available GB after acting."""
    a0 = avail_gb()
    workers = fetch_workers()
    acted = []
    for p, par in workers:
        if p.pid in caps:
            continue
        try:
            rss = p.memory_info().rss / 2**30
        except Exception:
            continue
        ok, err = cap(p.pid, CAP_MIN)
        caps[p.pid] = CAP_MIN if ok else None
        acted.append(f"cap pid {p.pid} (rss {rss:.1f} GB, parent {par.pid}) -> {'%g GB' % CAP_MIN if ok else 'FAILED err %d' % err}")
    live = [p for p, _ in workers if caps.get(p.pid)]
    a1 = a0
    if live and a0 < FLOOR_GB + 1.5:
        deficit = (FLOOR_GB + 3.0) - a0
        for p in live:
            try:
                rss = p.memory_info().rss / 2**30
            except Exception:
                continue
            new = max(CAP_MIN, min(caps[p.pid], rss) - deficit / len(live))
            if new < caps[p.pid] - 0.25:
                ok, err = cap(p.pid, new)
                if ok:
                    caps[p.pid] = new
                acted.append(f"lower cap pid {p.pid} -> {new:.1f} GB (rss {rss:.1f}){'' if ok else ' FAILED err %d' % err}")
        time.sleep(3)
        a1 = avail_gb()
        if a1 < FLOOR_GB and time.time() - last_empty[0] > 60:
            for p in live:
                acted.append(f"empty pid {p.pid} -> {'ok' if empty(p.pid) else 'FAILED'}")
            last_empty[0] = time.time()
            time.sleep(3)
            a1 = avail_gb()
    elif live and a0 > TARGET_GB + 2.0:
        spare = a0 - TARGET_GB          # the caps may exceed the working sets by this much in total,
        for p in live:                  # so cap = rss + share: growth is bounded by the spare, not compounded
            try:
                rss = p.memory_info().rss / 2**30
            except Exception:
                continue
            new = min(CAP_MAX, max(caps[p.pid], rss + spare / len(live)))
            if new > caps[p.pid] + 0.5:
                ok, err = cap(p.pid, new)
                if ok:
                    caps[p.pid] = new
                acted.append(f"raise cap pid {p.pid} -> {new:.1f} GB{'' if ok else ' FAILED err %d' % err}")
    if acted:
        log(f"available {a0:.1f} -> {a1:.1f} GB; " + "; ".join(acted))
    return a1


def main():
    caps, last_empty = {}, [0.0]
    if "--once" not in sys.argv:
        # singleton: the logon task and a hand-detached copy must not both run
        me, parent = os.getpid(), os.getppid()   # the .venv launcher stub is our parent and shows the same command line
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if p.pid in (me, parent):
                    continue
                if (p.info["name"] or "").lower() == "python.exe" and "ram_guard.py" in cmdline(p) and "--once" not in cmdline(p):
                    log(f"another ram_guard.py is running (pid {p.pid}); this one (pid {me}) exits")
                    return 0
            except Exception:
                continue
    log(f"ram_guard start: cap {CAP_MIN:g}..{CAP_MAX:g} GB per fetch worker (dynamic), floor {FLOOR_GB:g} GB available, target {TARGET_GB:g}; pid {os.getpid()}")
    if "--once" in sys.argv:
        a = one_pass(caps, last_empty)
        log(f"once: available now {a:.1f} GB")
        return 0
    low_since = None
    while True:
        try:
            # forget workers that exited
            for pid in list(caps):
                if not psutil.pid_exists(pid):
                    del caps[pid]
            a = one_pass(caps, last_empty)
            if a < FLOOR_GB:
                if low_since is None:
                    low_since = time.time()
                elif time.time() - low_since > 300:
                    log(f"WARNING: available {a:.1f} GB below the {FLOOR_GB:g} GB floor for 5 min with no worker left to trim")
                    low_since = time.time()
            else:
                low_since = None
        except Exception as e:
            log(f"error: {e!r}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
