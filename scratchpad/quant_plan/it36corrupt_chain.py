"""Tenth chain: the CORRUPTION (finiteness) check on the 50 longest-history users.

Andrew's requirement (2026-11-21): "For each user, calculate the average card history length. Find 50
users with longest average review histories. Run the quantized model on them. If even ONE card/note
state has non-finite values, or if even ONE p(recall) is non-finite, it counts as failing the check.
And make sure the histories aren't truncated."

What is checked, and how it is possible at all (see corruption/finite_probe.py for the long version):
  * the RAW recurrent state (before `_sanitize_state` caps it), the token-shift vectors, and every
    p(recall) tensor of both heads, for every row of every one of the 50 users;
  * on the PYTHON QAT path (RWKV_QAT_FUSED=0, the documented bit-parity twin of the fused CUDA
    kernels), because that is the only place the state passes through Python;
  * the roster is `corruption/corrupt50.json`: the 50 longest-mean-history users of 5001-7500 whose
    FULL section fits the VRAM cap (Andrew: "skip users that cannot fit into VRAM" -> a 200,000
    section-row cap; the 62 users above it are recorded in that file's `excluded_too_big`).
  * histories are NOT truncated: the eval DB stores one section per user covering the full history,
    and this path adds no review cap (the 64/128-review cap is a training-chunker artifact).

Points (order = value, so a partial run still answers the important question first). Each point runs
`corruption/check_corruption.py --tag <name>` on ONE process (50 users, no sharding) and writes
`corruption/corruption_corrupt_<name>.json`:
  1. l2i8s4  - RESULTS row 17, the axis-1 best (card rank-2 int8 + int4 shifts, note rank-1 int8 + int4)
  2. l2i8    - row 15 (card rank-2 int8, note rank-1 int8)
  3. l3i8    - row 16 (card rank-3 int8, note rank-1 int8)
  4. l3i8s4  - row 18 (card rank-3 int8 + int4 shifts, note rank-1 int8 + int4)
  5. ctl_fp32- the SAME 124-step decay with quantization OFF (qat_it36ctl/ctl_124.pth): attribution
     only. If a quantized point fails and this one also fails, the cause is not quantization.

WAIT: this chain's own work does NOT match the sibling chains' BUSY_PAT, so a chain queued BEHIND it
must add `probe_get_result.py|check_corruption.py` to its pattern, or it will start on top of this one.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
COR = RUN / "corruption"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36corrupt_chain.log"
LOCK = RUN / "it36corrupt_chain.lock"
PREV_LOG = RUN / "it36longdecay_chain.log"
PREV_END = "=== it36longdecay chain end ==="
DRIVER = COR / "check_corruption.py"
# Real GPU work only (the deadlock lesson: never list a sibling chain script, its waiting process
# would count as busy forever and no chain would ever take the GPU).
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|"
            "rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 900
IDLE_PROCEED_MIN = 12
HEARTBEAT_MIN = 5

POINTS = [
    ("l2i8s4", "RESULTS row 17, axis-1 best"),
    ("l2i8", "RESULTS row 15"),
    ("l3i8", "RESULTS row 16"),
    ("l3i8s4", "RESULTS row 18"),
    ("ctl_fp32", "same decay, quantization OFF (attribution)"),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


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


def wait_for_prev_chain():
    t0 = time.time()
    idle = 0
    while True:
        ended = PREV_END in PREV_LOG.read_text(encoding="utf-8") if PREV_LOG.exists() else False
        busy = running(BUSY_PAT)
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


sys.path.insert(0, str(RUN))
import it36ladder_lib as _lib  # noqa: E402  the shared wedge watchdog + retry (2026-11-22)


def run(tag, cmd, logfile):
    """Run one point through the shared watchdog (`it36ladder_lib.run`).

    That watchdog logs a heartbeat every 5 minutes AND kills the child if it writes nothing and burns
    no CPU for 20 minutes (the 2026-11-22 02:33 wedge: an F: HDD I/O error, events 153/51, left the
    trainer and its fetch worker at 0 CPU for 8.7 h while the chain log looked merely "busy"). The
    watchdog's presence also keeps THIS chain log fresh, so the hourly supervisor's "log unchanged >
    45 min" rule cannot mistake a healthy 50-user corruption point for a dead chain.
    """
    return _lib.run(log, tag, cmd, logfile)


def summary_block():
    """One line per finished point, read back from the driver's JSON, into the chain log."""
    for name, note in POINTS:
        p = COR / f"corruption_corrupt_{name}.json"
        if not p.exists():
            log(f"  {name:9s} (no result yet)")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ll = d.get("mean_logloss") or {}
        log(f"  {name:9s} {d['status']:10s} state_calls={d['state_calls']:,} "
            f"state_bad={d['state_bad']} p_bad={d['p_bad']} "
            f"scored={d['users_scored']}/{d['users_requested']} "
            f"skips={d.get('skips')} logloss ahead={ll.get('ahead')} imm={ll.get('imm')} "
            f"({d['secs']/60:.1f} min) -- {note}")


def main():
    log("=== it36corrupt chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36corrupt chain end (without running anything) ===")
        return 1

    for name, note in POINTS:
        out = COR / f"corruption_corrupt_{name}.json"
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS":
                    log(f"{name}: PASS already recorded -- skipping")
                    continue
            except (KeyError, ValueError):
                pass
        log(f"--- point {name} ({note})")
        rc = run(f"corrupt {name}",
                 [str(PY), "-u", str(DRIVER), "--tag", name, "--device", "cuda"],
                 COR / f"launcher_{name}.log")
        if rc != 0:
            log(f"{name}: driver rc={rc} (a FAIL or INCOMPLETE verdict is recorded in its JSON)")
        summary_block()

    log("=== it36corrupt chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
