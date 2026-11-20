"""Second smaller-state screening chain on the iter36 anchor (2026-11-20 23:50).

Continues `it36qatsmall_chain.py` after it ends, one GPU job at a time. Same gate as everything else on
this arm: every quantized point is judged against the SAME run with quantization OFF (`qnj_it36ctl`
for 124-step points, `qnj_it36ctl496` - already scored, 0.321659/0.294614 - for 496-step points).

Why these points:

1. `l1i4` at 496 steps (tag `it36l1i4496`), against the 496-step twin. Row 0b showed the decay length
   moves the loss ~25x more than the whole tax, and row 5 showed `l1i4` costs +0.001050/+0.001682 at
   124 steps. This point asks whether the longer decay absorbs that cost - i.e. whether the CHEAP
   allocation can be made compliant at all.
2. `l1i8s4` and `l1i4s8` at 124 steps (tags `it36small8s4`, `it36small4s8`): both are 768 card /
   2,304 note bits, with the bits moved from the rank-1 factors to the token-shift vectors or back, so
   the pair says which side of the rank-1 split carries the tax (rows 9-10 of RESULTS.md).
3. `l1i4s2` and `l1i2` at 496 steps: the same "does decay length absorb the tax" question at 384/1,152
   and 256/768 bits. The `l1i2` point is skipped unless its 124-step deltas are within +0.0025, because
   a catastrophic point will not be rescued by a longer decay.

Every point is relaunch-safe: a point whose merged result already exists is skipped whole.

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatsmall2_chain.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatsmall2_chain.log"
LOCK = RUN / "it36qatsmall2_chain.lock"
PREV_LOG = RUN / "it36qatsmall_chain.log"
PREV_END = "=== it36qatsmall chain end ==="
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
CTL_496 = "qnj_it36ctl496"
# Deliberately does NOT match this file: the VBS/PowerShell wrapper around this process is invisible to
# the chain loops, so the chain must not count itself as GPU work. `it36qatsmall_chain.py` is the
# previous chain (whose end marker this one waits for).
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|"
            "rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 480
IDLE_PROCEED_MIN = 12  # previous chain unseen this long with no GPU work => it is gone, proceed

sys.path.insert(0, str(RUN))


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
    """Wait until the previous chain has ended AND nothing on the GPU is left running."""
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


def eval_checkpoint(label, ckpt, ref, cand, stamp):
    tag = f"q{cand}_{stamp}"
    if (RESULT / f"adw_{tag}.jsonl").exists():
        log(f"{label}: {tag} already exists -- skipping")
        return 0
    return run(label, [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                       "--ref", ref, "--candidates", cand, "--shards", "2", "--solo", "1000000",
                       "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def qat_point(cand, steps, epochs, stamp, ref):
    """0.04-epoch smoke -> QAT decay from iter36 WS-final -> quantized eval against its own twin."""
    if (RESULT / f"adw_q{cand}_{stamp}.jsonl").exists():
        log(f"{cand}@{steps}: result exists -- skipping the whole point")
        return 0
    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", cand,
            "--train-users", "1", "100", "--max-train-len", "66000",
            "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558"]
    rc = run(f"smoke {cand}@{steps}",
             base + ["--epochs", "0.04", "--max-steps", "6", "--tag", f"{stamp}smoke"],
             RUN / f"{stamp}smoke_launcher.log")
    if rc != 0:
        log(f"smoke FAILED for {cand}@{steps} -- skipping this point")
        return rc
    rc = run(f"qat {cand}@{steps}",
             base + ["--epochs", str(epochs), "--max-steps", str(steps), "--prefix", "qat",
                     "--tag", stamp], RUN / f"{stamp}_launcher.log")
    if rc != 0:
        log(f"QAT FAILED for {cand}@{steps} -- skipping this point")
        return rc
    ckpt = RUN / f"qat_{stamp}" / f"qat_{steps}.pth"
    if not ckpt.exists():
        log(f"checkpoint missing: {ckpt}")
        return 1
    return eval_checkpoint(f"eval {cand}@{steps}", ckpt, ref, cand, stamp)


def delta_ahead(tag, ref):
    """Paired mean ahead delta of `tag` against `ref`, or None if either result is missing."""
    import quant_sweep as qs

    try:
        cur = qs.load_pair(tag)
        base = qs.load_pair(ref)
        if cur is None or base is None:
            return None
        return qs.paired(base, cur, 0)["delta"]
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot decide", so skip the point
        log(f"could not compute the delta for {tag} vs {ref}: {exc}")
        return None


def main():
    log("=== it36qatsmall2 chain start ===")
    # This chain is deliberately NOT in BUSY_PAT, because it must not count itself as GPU work. A lock
    # file guards against a double launch instead: two copies would both start the same QAT point and
    # clobber each other's checkpoints. Delete the lock by hand before relaunching after an outage.
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36qatsmall2 chain end (without running anything) ===")
        return 1

    # 1. the cheapest allocation that is still under both bars, at the longer decay
    qat_point("l1i4", 496, 8, "it36l1i4496", CTL_496)

    # 2. attribution at a fixed 768/2,304 bits: factors-heavy vs shifts-heavy
    qat_point("l1i8s4", 124, 2, "it36small8s4", CTL_124)
    qat_point("l1i4s8", 124, 2, "it36small4s8", CTL_124)

    # 3. the same decay-length question at the smaller allocations
    qat_point("l1i4s2", 496, 8, "it36l1i4s2496", CTL_496)
    d = delta_ahead("ql1i2_it36small2", CTL_124)
    if d is None:
        log("l1i2@496: the 124-step l1i2 result is not available -- skipping")
    elif d > 0.0025:
        log(f"l1i2@496: the 124-step l1i2 point is {d:+.6f} ahead (over +0.0025) -- skipping, a "
            f"longer decay will not rescue it")
    else:
        qat_point("l1i2", 496, 8, "it36l1i2496", CTL_496)

    log("=== it36qatsmall2 chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
