"""Is the GPU actually free? Two witnesses, and a 2-minute average -- never an instant reading.

WHY NOT `nvidia-smi` ONCE. An instantaneous utilization figure is worthless here, and this
project has already been burned by reading one: on 2026-08-25 the GPU read 0% while Andrew's
benchmark was very much mid-run, simply between CUDA phases. Andrew's fix: average over
2 minutes.

WHY TWO WITNESSES. The record's own rule -- "an alert built on a single witness reports the
witness's health, not the system's". Average utilization alone still misreads a long CPU phase
of a GPU job as "free". So this also asks whether any heavy long-lived python process is alive:
Andrew's srs-benchmark runs accumulate tens of CPU-hours and are the thing we must not collide
with. FREE requires BOTH: low average utilization AND no such process.

⚠ Do NOT kill anything this reports. The benchmark pythons are Andrew's, and CLAUDE.md is
explicit about it.

Usage: .venv/Scripts/python.exe scratchpad/gpu_free.py [seconds]     (default 120)
Exit code 0 = free, 1 = busy -- so a runner can gate on it.

⚠ DO NOT PIPE IT. `gpu_free.py | tail` reports TAIL's exit status, not the verdict --
which read as 'free' on the first run while the script itself printed BUSY. Run it bare and
read the RESULT line, or use its status directly (`if python gpu_free.py; then ...`).
"""
import subprocess
import sys
import time

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
INTERVAL = 2.0
# A GPU training job of ours sits far above this even between steps; desktop compositing sits
# far below. Chosen to be unambiguous rather than tight.
UTIL_FREE_PCT = 8.0
# CPU-seconds above which a python process is "a real job", not a helper script.
HEAVY_CPU_S = 3600.0


def sample_util(seconds):
    vals = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            u, m = out.split(",")[:2]
            vals.append((float(u), float(m)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(INTERVAL)
    return vals


def heavy_pythons():
    """Long-lived python processes with substantial accumulated CPU time."""
    ps = (
        "Get-Process python,pythonw -EA SilentlyContinue | "
        "ForEach-Object { '{0}|{1}|{2}' -f $_.Id, [int]$_.CPU, $_.StartTime }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        pid, cpu, started = int(parts[0]), float(parts[1]), parts[2]
        if cpu >= HEAVY_CPU_S:
            rows.append((pid, cpu, started))
    return rows


print("sampling GPU for %ds ..." % SECS, flush=True)
vals = sample_util(SECS)
if not vals:
    print("RESULT: UNKNOWN -- nvidia-smi produced no samples")
    sys.exit(1)

us = [v[0] for v in vals]
ms = [v[1] for v in vals]
avg, peak = sum(us) / len(us), max(us)
print("  utilization over %d samples: mean %.1f%%  peak %.0f%%" % (len(us), avg, peak))
print("  memory used: mean %.0f MiB  peak %.0f MiB" % (sum(ms) / len(ms), max(ms)))

heavy = heavy_pythons()
print("  heavy python processes (>= %.0f CPU-s):" % HEAVY_CPU_S)
for pid, cpu, started in heavy:
    print("     pid %-6d %8.0f CPU-s  started %s" % (pid, cpu, started))
if not heavy:
    print("     none")

util_ok = avg < UTIL_FREE_PCT
free = util_ok and not heavy
print("")
print("RESULT: %s   (avg util %s %.0f%%, heavy pythons: %d)"
      % ("FREE" if free else "BUSY",
         "<" if util_ok else ">=", UTIL_FREE_PCT, len(heavy)))
if not free and util_ok:
    print("  -> utilization is low but a long-running job is alive; that is a CPU phase,")
    print("     not an idle GPU. This is exactly the case an instant reading gets wrong.")
sys.exit(0 if free else 1)
