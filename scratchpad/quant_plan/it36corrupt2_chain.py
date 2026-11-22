"""Chain 12: the corruption sweep for the LATER rows - every deploy candidate gets its verdict.

`it36corrupt` (chain 10) covers the rows that passed when the column was added (rows 0 and 15-18).
Chains 11 and 13 produce new rows, and a row is deployable only if it ALSO has a corruption verdict
(Andrew's gate: even ONE non-finite card/note state value or even ONE non-finite p(recall) fails it).
This chain closes that gap, cheaply: a 50-user corruption run is minutes on the GPU, not hours.

It runs AFTER `it36combo` (chain 13), so every sweep it guards already exists. Each point is skipped
unless its own sweep verdict says the row PASSES the log-loss gate (`ub95 <= +0.0005` on BOTH heads):
a row that fails the log-loss gate is not a deploy candidate and does not need the check. A missing
sweep verdict is also a skip, and is logged, so "no corruption verdict" never looks like a pass.

Points (name = the corruption config tag in `corruption/check_corruption.py`):
  l2i4s4       RESULTS row 23            (card rank-2 int4)
  l3i4s4       chain 11 point 1          (card rank-3 int4)
  l2i8s2       chain 11 point 2          (card int2 shifts)
  n1i8s3       chain 11 point 3          (note int3 shifts)
  n2i4s3       chain 11 point 4          (note rank-2 int4 + int3 shifts)
  l3i4s4n8s3   chain 13 point 1          (cheap card under cheap note)
  l3i4s4n3     chain 13 point 2
  l2i4s4n8s3   chain 13 point 3

Its own driver matches `check_corruption`, so BUSY_PAT carries the corruption pattern too (the queue
warning): a chain queued behind this one must also see corruption work as GPU work.
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
LOG = RUN / "it36corrupt2_chain.log"
LOCK = RUN / "it36corrupt2_chain.lock"
PREV_LOG = RUN / "it36combo_chain.log"
PREV_END = "=== it36combo chain end ==="
DRIVER = COR / "check_corruption.py"
BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
WAIT_MAX_MIN = 900
IDLE_PROCEED_MIN = 12
HEARTBEAT_MIN = 5
GATE_UB95 = 0.0005

# (corruption config name, the sweep stamp that owns the gate, what the row is)
POINTS = [
    ("l2i4s4", "it36l2i4s4", "RESULTS row 23 (card rank-2 int4)"),
    ("l3i4s4", "it36l3i4s4", "chain 11 point 1 (card rank-3 int4)"),
    ("l2i8s2", "it36l2i8s2", "chain 11 point 2 (card int2 shifts)"),
    ("n1i8s3", "it36n1i8s3", "chain 11 point 3 (note int3 shifts)"),
    ("n2i4s3", "it36n2i4s3", "chain 11 point 4 (note rank-2 int4 + int3 shifts)"),
    ("l3i4s4n8s3", "it36l3i4s4n8s3", "chain 13 point 1 (cheap card under cheap note)"),
    ("l3i4s4n3", "it36l3i4s4n3", "chain 13 point 2"),
    ("l2i4s4n8s3", "it36l2i4s4n8s3", "chain 13 point 3"),
    ("l3i4s4n4", "it36l3i4s4n4", "shrink point 1 (card rank-3 int4 + note int4)"),
    ("l2i8s4n3", "it36l2i8s4n3", "shrink point 2 (note rank-1 int3 factors)"),
    ("l2i8s2n4", "it36l2i8s2n4", "shrink point 3 (card int2 shifts + note int4)"),
    ("l3i8s4n4s2", "it36l3i8s4n4s2", "shrink point 4 (note int4 + int2 shifts)"),
    ("l2i8n2i8", "it36l2i8n2i8", "chain 14 point 1 (note rank-2 int8 + int4 shifts)"),
    ("l3i8n2i8", "it36l3i8n2i8", "chain 14 point 2 (rank-3 card + note rank-2 int8)"),
    ("l2i8n2i8s8", "it36l2i8n2i8s8", "chain 14 point 3 (note rank-2 int8 + int8 shifts)"),
    ("l2i8n3i8", "it36l2i8n3i8", "chain 14 point 4 (note rank-3 int8)"),
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

    A heartbeat every 5 minutes, and a kill when the child writes nothing and burns no CPU for 20
    minutes (the 2026-11-22 02:33 F: HDD I/O-error wedge, 8.7 h at 0 CPU). Both matter here: this
    chain's points are the ones that would otherwise sit inside a corruption eval for a long time.
    """
    return _lib.run(log, tag, cmd, logfile)


def gate_passed(stamp):
    """True / False / None (no verdict yet). A row is a deploy candidate only on PASS of both heads."""
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text(encoding="utf-8")).get("rows") or []
    except ValueError:
        return None
    if not rows:
        return None
    g = rows[0].get("gate") or {}
    a = (g.get("ahead") or {}).get("ub95")
    i = (g.get("imm") or {}).get("ub95")
    if a is None or i is None:
        return None
    return bool(a <= GATE_UB95 and i <= GATE_UB95)


def summary_block():
    for name, stamp, note in POINTS:
        p = COR / f"corruption_corrupt_{name}.json"
        if not p.exists():
            log(f"  {name:12s} (no verdict)")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ll = d.get("mean_logloss") or {}
        log(f"  {name:12s} {d['status']:10s} state_calls={d['state_calls']:,} "
            f"state_bad={d['state_bad']} p_bad={d['p_bad']} "
            f"scored={d['users_scored']}/{d['users_requested']} skips={d.get('skips')} "
            f"logloss ahead={ll.get('ahead')} imm={ll.get('imm')} "
            f"({d['secs']/60:.1f} min) -- {note}")


def main():
    log("=== it36corrupt2 chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36corrupt2 chain end (without running anything) ===")
        return 1

    for name, stamp, note in POINTS:
        out = COR / f"corruption_corrupt_{name}.json"
        if out.exists():
            try:
                if json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS":
                    log(f"{name}: PASS already recorded -- skipping")
                    continue
            except (KeyError, ValueError):
                pass
        ok = gate_passed(stamp)
        if ok is False:
            log(f"{name}: {stamp} fails the log-loss gate -- not a deploy candidate, no check needed")
            continue
        if ok is None:
            log(f"{name}: no usable sweep verdict for {stamp} yet -- skipping "
                f"(a missing verdict is never a pass)")
            continue
        if not (RUN / f"qat_{stamp}" / "qat_124.pth").exists():
            log(f"{name}: checkpoint missing for {stamp} -- skipping")
            continue
        log(f"--- point {name} ({note}) passes the gate: running the corruption check")
        rc = run(f"corrupt {name}",
                 [str(PY), "-u", str(DRIVER), "--tag", name, "--device", "cuda"],
                 COR / f"launcher_{name}.log")
        if rc != 0:
            log(f"{name}: driver rc={rc} (a FAIL or INCOMPLETE verdict is in its JSON)")
        summary_block()

    log("=== it36corrupt2 chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
