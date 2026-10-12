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
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX_FILE = os.path.join(REPO, "scratchpad", "cloud", "BOX.local.md")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--no-box", action="store_true")
    args = ap.parse_args()
    cold = []

    running, newest, age, unfinished = local_state(args.minutes)
    print("LOCAL : trainer/eval process %s | newest run log %s (%.0f min old) | in flight: %s"
          % ("RUNNING" if running else "none", os.path.basename(newest or "-"), age, unfinished or "none"))
    if not running and age > args.minutes and unfinished:
        cold.append("LOCAL: no trainer, no log written for %.0f min, and %s has no terminal marker"
                    % (age, unfinished[0]))

    local_idle = (not running) and age > args.minutes
    # fault injection, default off: proves the LOCAL-idle clause below can actually fire.
    # A guard nobody has seen fail is a guard nobody has tested (RWKV_EVAL_FAKE_OOM's idiom).
    if os.environ.get('RWKV_COLD_FAKE_LOCAL_IDLE') == '1':
        local_idle = True
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
