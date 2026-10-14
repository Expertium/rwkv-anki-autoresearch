"""Has either GPU gone COLD with work still outstanding? (Andrew 2026-09-20)

Written after the rented box sat idle for 3.7 hours: both of its lanes had stopped on guards that were
wrong, and nothing told anyone. A run that ENDS is not an incident; a GPU that is idle while work is
outstanding is.

COLD means all three, per machine:
  1. no compute process on the GPU (locally: no python running train_rwkv / get_result / eval_sharded),
  2. no run log written for MORE than --minutes (default 30) -- progress, not utilization, because a
     memory-bound giant user can sit at 50 W and 100% for minutes and a stalled job can sit at 0%,
  3. work is still OUTSTANDING: a run log without a terminal DONE_EXIT_ marker, or a queued tag whose
     log does not end DONE_EXIT_0.
Rule 3 is what keeps it quiet when a queue has legitimately finished.

Exit 0 = both warm (or done). Exit 2 = at least one machine is COLD; the reason is printed.
`--no-box` skips the box (no ssh). Usage: python scratchpad/gpu_cold_check.py [--minutes 30] [--no-box]
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX_FILE = os.path.join(REPO, "scratchpad", "cloud", "BOX.local.md")
BOARD_FILE = os.path.join(REPO, "optimization", "GPU_RUN_BOARD.md")
BOARD_PATTERNS = {
    "d128e10 evaluation": r"\bd128e10\b",
    "architecture-agnostic speedup": r"Architecture-agnostic speedup",
    "Muon LR trials": r"wtmuon0p5|wtmuon2",
    "warmup trial": r"wtwarmup2000",
}


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip()
    except Exception as exc:                                  # a failed probe is not a verdict
        return "PROBE_ERROR %s" % exc


def box_ssh():
    """The ssh line lives only in the gitignored BOX.local.md -- the address stays out of git."""
    if not os.path.exists(BOX_FILE):
        return None
    m = re.search(r"ssh -p \d+ root@[\w.\-]+", io.open(BOX_FILE, encoding="utf-8").read())
    return m.group(0) if m else None


def local_state(minutes):
    busy = sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \\"name=\'python.exe\'\\" | '
              'Where-Object { $_.CommandLine -match \'train_rwkv|get_result|eval_sharded\' }).Count"')
    running = busy.isdigit() and int(busy) > 0
    newest, age, unfinished = None, 1e9, []
    for f in glob.glob(os.path.join(REPO, "scratchpad", "*", "*.log")):
        base = os.path.basename(f)
        # waiter logs never carry a terminal marker when a waiter was stopped by hand with a note, so they
        # would make every quiet local GPU look cold. Only RUN logs answer "is something in flight".
        if base.startswith("wait_") or ".cascade" in base:
            continue
        a = (time.time() - os.path.getmtime(f)) / 60.0
        if a < age:
            newest, age = f, a
        # A run is "in flight" only while its log is RECENT. A cancelled run (the 2 x b14 dry twin,
        # stopped when the ladder picked 3 x b10) never gets a marker, and after this window it must
        # stop implying that something should be on the GPU. A run that dies still alarms for 6 hours.
        if a < 60 * 6:
            tail = io.open(f, errors="replace").read()[-4000:]
            if "===== " in tail and "DONE_EXIT_" not in tail:
                unfinished.append(os.path.basename(f))
    return running, newest, age, unfinished
def local_queued_work():
    """Return unfinished work listed on the local-first GPU board."""
    try:
        with io.open(BOARD_FILE, encoding="utf-8") as f:
            board = f.read()
    except OSError as exc:
        return None, ["GPU run board unreadable: %s" % exc]
    active = re.search(r"## Active and queued runs\s*(.*?)\nThe chain files are", board, re.S)
    upcoming = re.search(r"## Next work after d128e10, before the queued WS trials\s*(.*?)\n## Later work", board, re.S)
    if not active or not upcoming:
        return None, ["GPU run board format unrecognized"]
    rows, next_work = active.group(1), upcoming.group(1)
    queued = []
    def board_row(pattern, section):
        for line in section.splitlines():
            if re.search(pattern, line, re.I):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) >= 4 and re.search(r"\b(queued|next|planned|running)\b", cells[2], re.I):
                    return True
        return False
    eval_log = os.path.join(REPO, "scratchpad", "d128e10", "evalonly.log")
    if board_row(BOARD_PATTERNS["d128e10 evaluation"], rows):
        if not os.path.exists(eval_log) or "EVAL_OK" not in io.open(eval_log, encoding="utf-8", errors="replace").read():
            queued.append("d128e10 evaluation")
    speed_section = rows + "\n" + next_work
    if re.search(BOARD_PATTERNS["architecture-agnostic speedup"], speed_section) and (
        board_row(BOARD_PATTERNS["architecture-agnostic speedup"], rows)
        or re.search(r"\*\*\(Recommended\)\*\*.*Architecture-agnostic speedup", speed_section, re.I)
    ):
        speed_log = os.path.join(REPO, "scratchpad", "dispatch", "speedup_after_d128.log")
        if not os.path.exists(speed_log) or not re.search(r"^DONE_EXIT_0\b", io.open(speed_log, encoding="utf-8", errors="replace").read()[-4000:], re.M):
            queued.append("architecture-agnostic speedup")
    for pattern, label, run in ((BOARD_PATTERNS["Muon LR trials"], "Muon LR trials", "wtmuon0p5"),
                                (BOARD_PATTERNS["warmup trial"], "warmup trial", "wtwarmup2000")):
        if board_row(pattern, rows):
            run_log = os.path.join(REPO, "scratchpad", run, run + ".log")
            tail = io.open(run_log, encoding="utf-8", errors="replace").read()[-4000:] if os.path.exists(run_log) else ""
            if not re.search(r"^DONE_EXIT_0\\b", tail, re.M):
                queued.append(label)
    return board, queued


def local_compute_processes():
    out = sh("powershell -NoProfile -Command \"Get-CimInstance Win32_Process -Filter 'name=python.exe' | Where-Object { $_.CommandLine -match 'train_rwkv|get_result|eval_sharded' } | Select-Object -ExpandProperty ProcessId\"")
    return {int(x) for x in out.splitlines() if x.strip().isdigit()}





def local_cuda_utilization():
    """Return maximum CUDA-engine utilization, or None when Windows cannot sample it."""
    cmd = (
        "Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 | "
        "ForEach-Object { $_.CounterSamples | Where-Object { $_.InstanceName -match 'engtype_Cuda' } | "
        "ForEach-Object { '{0}|{1}' -f $_.InstanceName, "
        "([double]$_.CookedValue).ToString([cultureinfo]::InvariantCulture) } }"
    )
    out = sh('powershell -NoProfile -Command "' + cmd + '"', timeout=10)
    values, pids = [], set()
    for line in out.splitlines():
        match = re.search(r"pid_(\d+).*\|([0-9.]+)$", line.strip(), re.I)
        if match:
            value = float(match.group(2))
            values.append(value)
            if value > 0:
                pids.add(match.group(1))
    # Several CUDA contexts can share an engine, so use the maximum, not the sum.
    return (max(values), sorted(pids)) if values else (None, [])


def low_util_sample_state(state, utilization, threshold, now, minutes, running):
    """Update low-util duration from one valid sample; invalid telemetry breaks continuity."""
    if not running or utilization is None or utilization >= threshold:
        return {"low_since": None, "last_sample": None, "alerted": False}, 0.0, False
    if not state.get("low_since") or now - state.get("last_sample", 0) > max(600, minutes * 120):
        state = {"low_since": now, "alerted": False}
    state["last_sample"] = now
    duration = max(0.0, now - state["low_since"])
    alert = duration >= minutes * 60 and not state.get("alerted")
    if alert:
        state["alerted"] = True
    return state, duration, alert


def low_util_state(threshold, minutes, running):
    """Track sustained low CUDA utilization only while a training/eval process exists."""
    state_path = os.path.join(REPO, "scratchpad", "gpu_low_util_state.json")
    utilization, active_pids = local_cuda_utilization()
    try:
        state = json.load(io.open(state_path, encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    state, duration, alert = low_util_sample_state(state, utilization, threshold, time.time(), minutes, running)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    return utilization, active_pids, duration, alert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--no-box", action="store_true")
    ap.add_argument("--low-util-threshold", type=float, default=40.0)
    ap.add_argument("--low-util-minutes", type=float, default=20.0)
    args = ap.parse_args()
    cold = []

    running, newest, age, unfinished = local_state(args.minutes)
    board, queued = local_queued_work()
    if board is None:
        cold.extend("LOCAL: " + item for item in queued)
    util, active_pids, low_for, low_alert = low_util_state(args.low_util_threshold, args.low_util_minutes, running)
    util_text = "unavailable" if util is None else "%.1f%%" % util
    print("LOCAL CUDA: %s | training/eval=%s | low for %.1f min (alert below %.1f%% for %.1f min) | active counter PIDs: %s"
          % (util_text, running, low_for / 60, args.low_util_threshold, args.low_util_minutes, active_pids or "none"))
    if low_alert:
        cold.append("LOCAL: CUDA utilization stayed below %.1f%% for %.0f min while training/eval was running"
                    % (args.low_util_threshold, args.low_util_minutes))
    if os.environ.get("RWKV_COLD_FAKE_LOW_UTIL") == "1":
        cold.append("LOCAL: TEST alert for sustained low CUDA utilization")

    print("LOCAL : trainer/eval process %s | newest run log %s (%.0f min old) | in flight: %s"
          % ("RUNNING" if running else "none", os.path.basename(newest or "-"), age, unfinished or "none"))
    if not running and age > args.minutes and unfinished:
        cold.append("LOCAL: no trainer, no log written for %.0f min, and %s has no terminal marker"
                    % (age, unfinished[0]))

    local_idle = (not running) and age > args.minutes
    if local_idle and queued:
        cold.append("LOCAL: GPU idle for %.0f min with queued work: %s" % (age, ", ".join(queued)))
    if os.environ.get("RWKV_COLD_FAKE_LOCAL_IDLE") == "1":
        local_idle = True
        if queued:
            cold.append("LOCAL: TEST alert with queued work: %s" % ", ".join(queued))
    retired = os.path.join(REPO, "scratchpad", "cloud", "BOX_RETIRED.txt")
    if not args.no_box and os.path.exists(retired):
        # 2026-09-29: the box's work ended and was archived; an idle box is then the expected state, not a
        # cold GPU. The file says when and why; delete it if a box runs work again.
        print("BOX   : retired (%s) -- skipped" % io.open(retired, encoding="utf-8").read().strip()[:120])
        args.no_box = True
    if not args.no_box:
        ssh = box_ssh()
        if not ssh:
            print("BOX   : no BOX.local.md -- skipped")
        else:
            out = sh(ssh + ' -o ConnectTimeout=20 -o BatchMode=yes "nvidia-smi --query-compute-apps=pid '
                           '--format=csv,noheader | wc -l; find /workspace/runs -name \'*.log\' -mmin -%d | wc -l; '
                           'for t in $(cat /workspace/queue_all.txt 2>/dev/null); do '
                           'tail -1 /workspace/runs/$t/$t.log 2>/dev/null | grep -q \'^DONE_EXIT_0 \' || echo $t; done"'
                     % int(args.minutes), timeout=90)
            lines = [l.strip() for l in out.split("\n") if l.strip() and "PROBE_ERROR" not in l
                     and "vast" not in l.lower() and "Have fun" not in l and "Welcome" not in l]
            if out.startswith("PROBE_ERROR") or len(lines) < 2:
                print("BOX   : UNREACHABLE (%s) -- not a verdict, check by hand" % out[:80])
            else:
                apps, fresh, pending = int(lines[0]), int(lines[1]), lines[2:]
                print("BOX   : %d compute app(s) | %d log(s) written in the last %d min | queued and unfinished: %s"
                      % (apps, fresh, int(args.minutes), pending or "none"))
                if apps == 0 and fresh == 0 and pending:
                    cold.append("BOX: GPU empty, no log for %d min, still queued: %s"
                                % (int(args.minutes), ", ".join(pending)))
                # ⚠⚠ THE CASE THIS CHECK MISSED ON 2026-09-21, and it is the one Andrew complained
                # about: the LOCAL card sat empty for four hours while nine arms were queued on the
                # box. Every clause above passed -- no local job had DIED, there simply was no local
                # job -- so the verdict read "both warm (or finished)". An idle GPU with work in
                # existence anywhere is the failure, whether or not that work was assigned locally.
                # A queued box arm can always be run here instead (cw10lf measured the cross-hardware
                # offset at a third of the noise floor), so "not assigned to this machine" is not a
                # reason for the machine to be empty.
                if local_idle and pending:
                    cold.append("LOCAL: GPU empty for %.0f min while %d arm(s) wait on the box (%s) "
                                "-- a queued arm can run here instead"
                                % (age, len(pending), ", ".join(pending[:4])))

    if cold:
        print("\nCOLD:")
        for c in cold:
            print("  * " + c)
        return 2
    print("\nboth warm (or finished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
