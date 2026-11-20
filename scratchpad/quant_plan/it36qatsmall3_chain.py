"""Third screening chain: per-stream attribution of row 4's tax (2026-11-21 01:00).

Waits for `it36qatsmall2_chain.py` to end, then runs the two per-stream twins of row 4's allocation at
124 steps against the same identical unquantized control (`qnj_it36ctl`):

  * `l1i8c` - card rank-1 int8 only, note left in fp32. Answers: how much of row 4's ahead tax
    (+0.000503) and imm tax (+0.001009) is the CARD stream's?
  * `l1i8n` - note rank-1 int8 only, card left in fp32. Answers the same for the NOTE stream.

Both are attribution rows: they deliberately leave one stream unquantized, so they do NOT meet
constraint 2 and are never candidates. They are the clean version of the decomposition that rows 3-4
only imply (c2_nl1i8 vs l1i8 differ in the card and have the same note, and their imm taxes are nearly
equal, which is what suggested the note owns the imm head).

They also tell us where a finer level would pay off. Within the rank-1 fast path the per-layer budget
(2,176 bits) still has headroom over row 4's 1,024 bits, but the only ways to spend it are finer levels
(int16, or unquantized factors) - a machinery change that needs Andrew's decision, so this chain stops
after the attribution pair and reports.

Relaunch-safe (a point whose result exists is skipped), lock-guarded, and it does not count itself as
GPU work (the venv `python.exe` is a shim, so one chain shows two processes).

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatsmall3_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatsmall3_chain.log"
LOCK = RUN / "it36qatsmall3_chain.lock"
PREV_LOG = RUN / "it36qatsmall2_chain.log"
PREV_END = "=== it36qatsmall2 chain end ==="
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
# Deliberately does NOT match this file; `it36qatsmall2_chain.py` is the chain whose end we wait for.
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|it36qatsmall2_chain.py|"
            "rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 480
IDLE_PROCEED_MIN = 12


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


def run(tag, cmd, logfile):
    log(f"START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def eval_point(cand, stamp, ckpt):
    """One quantized eval of a QAT checkpoint; the card/note twin is already trained."""
    tag = f"q{cand}_{stamp}"
    if (RESULT / f"adw_{tag}.jsonl").exists():
        log(f"{cand}: {tag} already exists -- skipping")
        return 0
    return run(f"eval {cand}",
               [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                "--ref", CTL_124, "--candidates", cand, "--shards", "2", "--solo", "1000000",
                "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def main():
    log("=== it36qatsmall3 chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36qatsmall3 chain end (without running anything) ===")
        return 1

    # The l1i8 124-step QAT checkpoint already exists: only the per-stream attribution evals are new.
    ckpt = RUN / "qat_it36qatl1i8" / "qat_124.pth"
    if not ckpt.exists():
        log(f"missing {ckpt} -- the attribution pair cannot be scored")
        log("=== it36qatsmall3 chain end ===")
        return 1
    eval_point("l1i8c", "it36attrc", ckpt)
    eval_point("l1i8n", "it36attrn", ckpt)

    log("=== it36qatsmall3 chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
