"""Diagnose the 8.7 h stall of the it36shrink l3i4s4n4 training (2026-11-22 11:15)."""
import pathlib
import subprocess
import time

import psutil

TRAIN_LOG = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan"
                         r"\qat_it36l3i4s4n4\train.log")
PARENT = 29396          # rwkv.train_rwkv
QW = 29396

print("=== process tree around pid", PARENT, "===")
procs = {}
for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
    try:
        cl = " ".join(p.info["cmdline"] or [])[:150]
    except Exception:
        cl = "?"
    procs[p.info["pid"]] = (p.info["ppid"], p.info["name"], cl)

def show(pid, indent=0):
    if pid not in procs:
        return
    ppid, name, cl = procs[pid]
    try:
        pr = psutil.Process(pid)
        rss = pr.memory_info().rss / 2**30
        cpu = pr.cpu_times()
        print(f"{'  ' * indent}pid {pid:6d} {name:12s} rss {rss:5.2f} GB "
              f"user {cpu.user:9.1f} sys {cpu.system:7.1f} threads {pr.num_threads():3d}  {cl[:110]}")
    except Exception as exc:
        print(f"{'  ' * indent}pid {pid} gone ({exc})")
    for child in [k for k, v in procs.items() if v[0] == pid]:
        show(child, indent + 1)

show(PARENT)

print("\n=== 6 s CPU delta on the parent + its children ===")
kids = [PARENT] + [k for k, v in procs.items() if v[0] == PARENT]
snap = {}
for pid in kids:
    try:
        snap[pid] = psutil.Process(pid).cpu_times()
    except Exception:
        pass
time.sleep(6)
for pid, t0 in snap.items():
    try:
        pr = psutil.Process(pid)
        t1 = pr.cpu_times()
        print(f"pid {pid:6d} user {t1.user - t0.user:7.3f} s  sys {t1.system - t0.system:6.3f} s "
              f"(in 6 s wall)  status {pr.status()}")
    except Exception as exc:
        print(f"pid {pid:6d} gone ({exc})")

print("\n=== train.log: mtime, size, last 12 lines ===")
st = TRAIN_LOG.stat()
print("mtime", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
      "size", st.st_size, "  (now", time.strftime("%H:%M:%S"), ")")
lines = TRAIN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
for l in lines[-12:]:
    print("   ", l[:160])

print("\n=== train.log: suspect lines ===")
for i, l in enumerate(lines, 1):
    low = l.lower()
    if any(k in low for k in ("traceback", "error", "exception", "worker", "exited", "died", "lost")):
        print(f"   {i}: {l[:160]}")
print("total lines:", len(lines))
