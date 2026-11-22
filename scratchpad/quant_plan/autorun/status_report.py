"""Deterministic state report for the hourly quant supervisor (autorun).

Reads files and the process list only. It never loads a model, never starts a run and never writes
anything, so it is safe to call at any time, including while a chain is training.

    .venv\\Scripts\\python.exe scratchpad\\quant_plan\\autorun\\status_report.py
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
RESULT = ROOT / "result"
RESULTS_MD = RUN / "RESULTS.md"
NOTES_MD = RUN / "NOTES.md"
RAM_GUARD = ROOT / "scratchpad" / "ram_guard.log"
RECORDED = HERE / "RECORDED.txt"
QUEUE = HERE / "QUEUE.md"
LOCK = HERE / ".autorun.lock"

# `probe_get_result|check_corruption` (2026-11-21): the corruption check runs a normal get_result
# eval under a different entry point, so without these two the status report would call the GPU IDLE
# while the it36corrupt chain is scoring, and the hourly supervisor could then restart that same
# chain on top of itself.
WORKLOAD = ("qat_decay|quant_sweep|eval_sharded|train_rwkv|rwkv-infer|rust_speed|rust_grid|"
            "eval_baseline|probe_get_result|check_corruption")
RAM_FLOOR_GB = 10.0
LOCK_STALE_MIN = 90
CHAIN_STALL_MIN = 45
# CPU a healthy workload process tree burns in a 5 s window. Added 2026-11-22 with cpu_used_s():
# the 8.7 h wedge (F: disk I/O error, System log events 153/51 at 02:33:45) sat at 0.000 s while the
# report still said "GPU BUSY" and the chain logs looked normal, so this reading is the only signal.
CPU_ALIVE_S = 0.05


def free_ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return None
    return ms.ullAvailPhys / 1024 ** 3


def ps(cmd):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                         capture_output=True, text=True)
    return out.stdout.strip()


def gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip() or "nvidia-smi gave no output"


def procs():
    raw = ps(f"Get-CimInstance Win32_Process | Where-Object {{ $_.Name -match 'python|rwkv-infer' "
             f"-and $_.CommandLine -match '{WORKLOAD}' }} | ForEach-Object "
             f"{{ \"$($_.ProcessId)|$($_.Name)|$($_.CommandLine)\" }}")
    rows = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()[:170]))
    return rows


def cpu_used_s(rows, seconds=5):
    """CPU seconds each workload process tree used over `seconds`.

    Added 2026-11-22 after the 8.7 h wedge: an F: HDD I/O error (System log events 153/51, 02:33:45)
    left `rwkv.train_rwkv` and its fetch worker at 0.000 s CPU while the report still said "GPU BUSY",
    so the hourly supervisor saw nothing wrong and the restart rule (which needs GPU IDLE) never fired.
    A 0.0 reading here is the wedge signal: healthy work always burns CPU.
    """
    try:
        import psutil
    except ImportError:
        return {}, set()
    before = {}
    for pid, _, _ in rows:
        try:
            pr = psutil.Process(int(pid))
            before[int(pid)] = tree_cpu(pr)
        except Exception:
            before[int(pid)] = None
    time.sleep(seconds)
    out = {}
    gone = set()
    for pid, base in before.items():
        try:
            now = tree_cpu(psutil.Process(pid))
        except Exception:
            out[pid] = None
            gone.add(pid)
            continue
        out[pid] = None if base is None else now - base
    return out, gone


def tree_cpu(pr):
    total = 0.0
    procs = [pr]
    try:
        procs += pr.children(recursive=True)
    except Exception:
        pass
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except Exception:
            pass
    return total


def chains():
    rows = []
    for p in sorted(RUN.glob("*_chain.log"), key=lambda q: q.stat().st_mtime, reverse=True):
        name = p.stem.replace("_chain", "")
        text = p.read_text(encoding="utf-8", errors="replace")
        last = next((l for l in reversed(text.splitlines()) if l.strip()), "(empty)")
        # exact marker, because "prev chain ended=False" also contains the substring "chain end"
        ended = f"=== {name} chain end" in text
        age = (time.time() - p.stat().st_mtime) / 60
        rows.append((name, last, ended, age))
    return rows


def guard_age_min():
    if not RAM_GUARD.exists():
        return None, "(no ram_guard.log)"
    last = next((l for l in reversed(RAM_GUARD.read_text(encoding="utf-8", errors="replace").splitlines())
                 if l.strip()), "")
    try:
        stamp = time.mktime(time.strptime(last[:19], "%Y-%m-%d %H:%M:%S"))
        return (time.time() - stamp) / 60, last
    except ValueError:
        return None, last


def recorded_stamps():
    if not RECORDED.exists():
        return set()
    return {l.strip() for l in RECORDED.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def f6(v):
    return "n/a" if v is None else f"{v:+.6f}"


def unrecorded():
    done = recorded_stamps()
    rows = []
    for p in sorted(RUN.glob("quant_sweep_*.json")):
        stamp = p.stem[len("quant_sweep_"):]
        if stamp in done:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # a half-written or broken json must not kill the report
            rows.append((stamp, f"UNREADABLE ({exc})"))
            continue
        for r in d.get("rows", []):
            g = r.get("gate") or {}
            a, i = g.get("ahead") or {}, g.get("imm") or {}
            b = r.get("bits") or {}
            ok = (a.get("ub95") is not None and a["ub95"] <= 0.0005
                  and i.get("ub95") is not None and i["ub95"] <= 0.0005)
            rows.append((stamp, f"{r.get('name')} card={b.get('card')} note={b.get('note')} "
                                f"bits_ok={r.get('bits_ok')} | ahead d={f6(a.get('delta'))} "
                                f"ub95={f6(a.get('ub95'))} | imm d={f6(i.get('delta'))} "
                                f"ub95={f6(i.get('ub95'))} | gate={'PASS' if ok else 'FAIL'} "
                                f"| secs={r.get('secs')}"))
    return rows


def queue_pending():
    """Real launch items only: a raw line that starts with the box marker. Indented examples in
    QUEUE.md are documentation, not items."""
    if not QUEUE.exists():
        return None
    return [l.strip() for l in QUEUE.read_text(encoding="utf-8").splitlines()
            if l.startswith("- [ ]")]


def lock_state():
    if not LOCK.exists():
        return "free"
    text = LOCK.read_text(encoding="utf-8", errors="replace").strip()
    age = (time.time() - LOCK.stat().st_mtime) / 60
    return f"HELD ({text}) age {age:.0f} min" + (" -- STALE" if age > LOCK_STALE_MIN else "")


def main():
    ram = free_ram_gb()
    print(f"=== AUTORUN STATUS {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"repo: {ROOT}")
    print(f"host RAM free: {'n/a' if ram is None else f'{ram:.1f}'} GB "
          f"(floor {RAM_FLOOR_GB:.0f}) -> {'OK' if ram and ram >= RAM_FLOOR_GB else 'LOW'}")
    print(f"GPU: {gpu()}")

    plist = procs()
    print(f"workload processes: {len(plist)} -> "
          f"{'GPU BUSY' if plist else 'GPU IDLE (no matching workload process)'}")
    for pid, name, cmd in plist:
        print(f"  pid {pid} {name}: {cmd}")

    if plist:
        used, gone = cpu_used_s(plist)
        shown = " ".join(f"{pid}={'gone' if int(pid) in gone or used.get(int(pid)) is None else f'{used[int(pid)]:.2f}s'}"
                         for pid, _, _ in plist)
        alive = [v for v in used.values() if v is not None and v > CPU_ALIVE_S]
        here = len(plist) - len(gone)
        if alive:
            verdict = f"CPU alive ({len(alive)}/{here} processes busy)"
        elif here == 0:
            # a chain that finished a point and started the next one between the two samples
            verdict = "all sampled processes exited during the sample (a point just changed)"
        else:
            verdict = "WEDGE? live workload processes burned no CPU - see section 2 of QUEUE.md"
        print(f"  CPU over 5 s: {shown} -> {verdict}")

    print("chain logs (newest first):")
    for name, last, ended, age in chains():
        state = "end marker present" if ended else "NO END MARKER"
        stall = " -- STALLED?" if (not ended and age > CHAIN_STALL_MIN) else ""
        print(f"  {name:14s} last line [{age:6.1f} min ago] {state}{stall}")
        print(f"      {last}")

    gage, gline = guard_age_min()
    if gage is None:
        print(f"ram guard: UNKNOWN | {gline}")
    else:
        print(f"ram guard: last heartbeat {gage:.1f} min ago -> "
              f"{'OK' if gage <= 10 else 'STALE'} | {gline}")

    print(f"locks: {lock_state()}")
    pend = queue_pending()
    print(f"QUEUE.md pending launch items: {'NO QUEUE.md' if pend is None else len(pend)}")
    for item in (pend or []):
        print(f"  {item}")

    rows = unrecorded()
    print(f"NOT RECORDED sweep results: {len(rows)}")
    for stamp, desc in rows:
        print(f"  {stamp:16s} {desc}")
    print("=== END AUTORUN STATUS ===")


if __name__ == "__main__":
    main()
