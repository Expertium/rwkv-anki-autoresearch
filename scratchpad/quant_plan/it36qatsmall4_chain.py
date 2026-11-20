"""Fourth screening chain: does a LONGER QAT decay reduce the tax? (2026-11-21 01:05)

Andrew relaxed the gate: the QAT decay may use more than the champion's 124 steps, accepting that the
result is then less apples-to-apples comparable, as long as a failing recipe can be turned into a winning
one. This chain tests the only untested direction that relaxation opens: a decay length ABOVE the 496
steps already measured.

What we know (all against the identical unquantized run at the same length):
  * 124 steps, `l1i8` (1,024/3,072 bits): tax +0.000503 ahead / +0.001009 imm.
  * 496 steps, `l1i8` (same bits):          tax +0.001132 ahead / +0.001674 imm - the tax GREW.
The tax curve therefore rises with the decay length over 124 -> 496 steps. This chain asks whether it
turns over later (992 steps = 2x the last point, 8x the champion's 124), with the fp32 twin trained at
the SAME length so the comparison stays honest - the twin is the only way to tell "quantization got
better" apart from "training got longer", and the latter also helps fp32.

Points, in order (each one skips itself if its merged result already exists):
  1. `ctl_992`: the 992-step fp32 decay (`--no-quant`), the identical twin for point 2, scored as `nj`
     against `qnj_it36_124` (context only) - its real job is to exist.
  2. `l1i8` at 992 steps, scored against `qnj_it36ctl992`.
  3. CONDITIONAL, only if the 496-step `l1i4` point beat the 124-step `l1i4` point on the ahead tax: the
     same 992-step test at the cheaper 512/1,536-bit allocation. If the longer decay did not help the
     cheapest allocation at 496, a 2x longer one is not worth 45 more minutes.

Relaunch-safe, lock-guarded, and it does not count itself as GPU work (the venv `python.exe` is a shim,
so one chain shows two processes). Delete the lock by hand after an outage.

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36qatsmall4_chain.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = RUN / "it36qatsmall4_chain.log"
LOCK = RUN / "it36qatsmall4_chain.lock"
PREV_LOG = RUN / "it36qatsmall3_chain.log"
PREV_END = "=== it36qatsmall3 chain end ==="
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
CTL_992 = "qnj_it36ctl992"
STEPS = 992
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|it36qatsmall2_chain.py|"
            "it36qatsmall3_chain.py|rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 600
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


def train_point(cand, stamp, control=False, smoke=True):
    """Smoke -> one decay of `STEPS` steps (fp32 control if `control`) -> return the checkpoint."""
    prefix = "ctl" if control else "qat"
    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", cand,
            "--train-users", "1", "100", "--max-train-len", "66000",
            "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558"]
    if control:
        base += ["--no-quant"]
    if smoke:
        rc = run(f"smoke {stamp}",
                 base + ["--epochs", "0.04", "--max-steps", "6", "--tag", f"{stamp}smoke"],
                 RUN / f"{stamp}smoke_launcher.log")
        if rc != 0:
            log(f"smoke FAILED for {stamp} -- skipping this point")
            return None
    rc = run(f"{prefix} {stamp}",
             base + ["--epochs", "8", "--max-steps", str(STEPS), "--prefix", prefix, "--tag", stamp],
             RUN / f"{stamp}_launcher.log")
    if rc != 0:
        log(f"{prefix} FAILED for {stamp} -- skipping this point")
        return None
    ckpt = RUN / f"qat_{stamp}" / f"{prefix}_{STEPS}.pth"
    if not ckpt.exists():
        log(f"checkpoint missing: {ckpt}")
        return None
    return ckpt


def eval_point(label, cand, stamp, ckpt, ref):
    tag = f"q{cand}_{stamp}"
    if (RESULT / f"adw_{tag}.jsonl").exists():
        log(f"{label}: {tag} already exists -- skipping")
        return 0
    return run(label, [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                       "--ref", ref, "--candidates", cand, "--shards", "2", "--solo", "1000000",
                       "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def ahead_tax(stamp, ref):
    """The ahead delta recorded by a finished sweep, or None."""
    p = RUN / f"quant_sweep_{stamp}.json"
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        return rows[0]["gate"]["ahead"]["delta"]
    except (KeyError, IndexError, ValueError) as exc:
        log(f"could not read the ahead delta from {p.name}: {exc}")
        return None


def main():
    log("=== it36qatsmall4 chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36qatsmall4 chain end (without running anything) ===")
        return 1

    # 1. the 992-step fp32 twin (identical recipe, quantization OFF)
    ctl = RUN / "qat_it36ctl992" / f"ctl_{STEPS}.pth"
    if not ctl.exists():
        ctl = train_point("nj", "it36ctl992", control=True)
        if ctl is None:
            log("=== it36qatsmall4 chain end (no twin, nothing to compare against) ===")
            return 1
    eval_point("eval ctl992", "nj", "it36ctl992", ctl, "qnj_it36_124")

    # 2. row 4's allocation at 992 steps, against that twin
    if not (RESULT / "adw_ql1i8_it36l1i8992.jsonl").exists():
        ck = train_point("l1i8", "it36l1i8992")
        if ck is not None:
            eval_point("eval l1i8@992", "l1i8", "it36l1i8992", ck, CTL_992)
    else:
        log("l1i8@992: result exists -- skipping the whole point")

    # 3. the cheap allocation at 992 steps, but only if 496 steps already helped it
    d496 = ahead_tax("it36l1i4496", "qnj_it36ctl496")
    d124 = ahead_tax("it36small4", CTL_124)
    if d496 is None or d124 is None:
        log("l1i4@992: cannot compare the 496-step and 124-step taxes -- skipping")
    elif d496 >= d124:
        log(f"l1i4@992: the 496-step tax {d496:+.6f} is not better than the 124-step tax {d124:+.6f} "
            f"-- a 2x longer decay is not worth it; skipping")
    else:
        ck = train_point("l1i4", "it36l1i4992")
        if ck is not None:
            eval_point("eval l1i4@992", "l1i4", "it36l1i4992", ck, CTL_992)

    log("=== it36qatsmall4 chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
