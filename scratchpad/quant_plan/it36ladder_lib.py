"""Shared helpers for the card/note state-quantization screening chains (2026-11-21 09:30).

One point = a 0.04-epoch smoke, one QAT decay of `steps` steps from the pre-decay champion checkpoint
(`pretrain/rwkv/opt_iter36/rwkv_iter36_558`), then the quantized eval of that candidate against a
reference tag. Points whose merged result already exists are skipped, so a relaunch is safe.

Chains are chained by END MARKERS: a chain waits until the previous chain's log holds its end marker AND
no matching process is running, then takes its own lock file. Each chain lives in its OWN script file,
because the venv `python.exe` is a shim (one chain shows two processes) and a chain must never count
itself as GPU work. Delete a stale lock by hand after an outage.
"""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ROSTER = RUN / "roster750_seed20261119.json"
START_FOLDER = "pretrain/rwkv/opt_iter36"
START_NAME = "rwkv_iter36_558"
WAIT_MAX_MIN = 900
IDLE_PROCEED_MIN = 12


def make_logger(path):
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def running(pattern):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -match 'python|rwkv-infer' -and "
         f"$_.CommandLine -match '{pattern}' }} | Measure-Object).Count"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def wait_for_prev(log, prev_log, prev_end, busy_pat):
    t0 = time.time()
    idle = 0
    while True:
        ended = prev_end in prev_log.read_text(encoding="utf-8") if prev_log.exists() else False
        busy = running(busy_pat)
        if ended and busy == 0:
            log("GPU free: previous chain logged its end marker, 0 matching processes")
            return True
        idle = idle + 1 if busy == 0 else 0
        if idle >= IDLE_PROCEED_MIN:
            log(f"no matching process for {idle} checks and no end marker -- assuming the previous "
                f"chain is gone; proceeding")
            return True
        if (time.time() - t0) / 60 > WAIT_MAX_MIN:
            log(f"GAVE UP waiting after {WAIT_MAX_MIN} min (ended={ended}, busy={busy})")
            return False
        time.sleep(60)
        log(f"waiting: prev chain ended={ended}, matching processes={busy}")


STALL_MIN = 20          # no child output AND no child CPU for this long = the run is wedged
STALL_RETRIES = 1       # a wedged run is retried once before the point is given up
POLL_S = 30
STALL_RC = -9

_PSUTIL = None


def _psutil():
    global _PSUTIL
    if _PSUTIL is None:
        import psutil as _p
        _PSUTIL = _p
    return _PSUTIL


def tree_cpu_s(pid):
    """CPU seconds used by pid and all its descendants, or None when the process is gone.

    The stall test needs BOTH halves: a wedged run writes nothing to its log AND burns no CPU (measured
    2026-11-22: 0.000 s over 6 s on both the trainer and its fetch worker). A healthy run may be quiet
    for a while (model load, a long validation pass) but its worker keeps consuming CPU, so a
    CPU-progress check is what stops the watchdog from killing good work.
    """
    ps = _psutil()
    try:
        me = ps.Process(pid)
        procs = [me] + me.children(recursive=True)
    except Exception:
        return None
    total = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except Exception:
            pass
    return total


def _run_once(log, tag, cmd, logfile):
    """One attempt: run the child, log a heartbeat, kill it if it wedges. STALL_RC = killed as wedged."""
    t0 = time.time()
    hb = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        last_size = logfile.stat().st_size if logfile.exists() else 0
        last_progress = time.time()
        last_cpu = tree_cpu_s(p.pid)
        while True:
            try:
                return p.wait(timeout=POLL_S)
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            size = logfile.stat().st_size if logfile.exists() else 0
            if size != last_size:
                last_size = size
                last_progress = now
            cpu = tree_cpu_s(p.pid)
            if cpu is not None and last_cpu is not None and cpu - last_cpu > 1.0:
                last_progress = now
            last_cpu = cpu
            if now - last_progress > STALL_MIN * 60:
                log(f"STALL {tag}: no log output and no CPU for {STALL_MIN} min -> killing pid "
                    f"{p.pid} with its children (this is the 2026-11-22 02:33 disk-error wedge: the "
                    f"trainer and its fetch worker sat at 0 CPU for 8.7 h)")
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True, text=True)
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    log(f"STALL {tag}: pid {p.pid} still alive 120 s after taskkill")
                return STALL_RC
            if now - hb >= 5 * 60:
                hb = now
                log(f"  ... {tag} running {(now - t0) / 60:.0f} min; log {size} bytes; "
                    f"cpu {cpu if cpu is None else f'{cpu:.1f}'} s")
    return -1


def run(log, tag, cmd, logfile):
    """Run one step of a point, with a wedge watchdog and one retry (the F: HDD throws I/O errors)."""
    for attempt in range(STALL_RETRIES + 1):
        suffix = f" (attempt {attempt + 1})" if attempt else ""
        log(f"START {tag}{suffix}: {' '.join(cmd)}")
        t0 = time.time()
        rc = _run_once(log, tag, cmd, logfile)
        log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
        if rc != STALL_RC:
            return rc
        log(f"{tag}: killed as wedged; "
            + ("retrying once" if attempt < STALL_RETRIES else "GIVING UP on this point"))
    return STALL_RC


def qat_point(log, cand, stamp, steps, ref, epochs):
    if (RESULT / f"adw_q{cand}_{stamp}.jsonl").exists():
        log(f"{cand}@{steps}: result exists -- skipping the whole point")
        return 0
    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", cand,
            "--train-users", "1", "100", "--max-train-len", "66000",
            "--start-folder", START_FOLDER, "--start-name", START_NAME]
    rc = run(log, f"smoke {cand}", base + ["--epochs", "0.04", "--max-steps", "6",
                                           "--tag", f"{stamp}smoke"],
             RUN / f"{stamp}smoke_launcher.log")
    if rc != 0:
        log(f"smoke FAILED for {cand} -- skipping this point")
        return rc
    rc = run(log, f"qat {cand}@{steps}", base + ["--epochs", str(epochs), "--max-steps", str(steps),
                                                 "--prefix", "qat", "--tag", stamp],
             RUN / f"{stamp}_launcher.log")
    if rc != 0:
        log(f"qat FAILED for {cand} -- skipping this point")
        return rc
    ckpt = RUN / f"qat_{stamp}" / f"qat_{steps}.pth"
    if not ckpt.exists():
        log(f"checkpoint missing: {ckpt}")
        return 1
    return run(log, f"eval {cand}@{steps}",
               [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                "--ref", ref, "--candidates", cand, "--shards", "2", "--solo", "1000000",
                "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def chain_main(name, prev_log_name, prev_end, points, busy_pat):
    """`points` = list of (candidate, stamp, steps, ref) tuples."""
    log = make_logger(RUN / f"{name}_chain.log")
    lock = RUN / f"{name}_chain.lock"
    prev_log = RUN / prev_log_name

    log(f"=== {name} chain start ===")
    if lock.exists():
        log(f"lock found ({lock.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {lock}")
    if not wait_for_prev(log, prev_log, prev_end, busy_pat):
        log(f"=== {name} chain end (without running anything) ===")
        return 1

    for cand, stamp, steps, ref in points:
        epochs = max(2, round(steps / 62))
        log(f"point {cand}@{steps} -> tag {stamp}, ref {ref}, epochs {epochs}")
        rc = qat_point(log, cand, stamp, steps, ref, epochs)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    log(f"=== {name} chain end ===")
    return 0
