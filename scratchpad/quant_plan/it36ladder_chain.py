"""Generic screening chain for the card/note state-quantization arm (2026-11-21 09:30).

One point = a 0.04-epoch smoke, one QAT decay of `--steps` steps from the pre-decay champion checkpoint
(`pretrain/rwkv/opt_iter36/rwkv_iter36_558`), then the quantized eval of that candidate against a
reference tag. Points are skipped when their merged result already exists, so a relaunch is safe.

Chains are chained by END MARKERS: this script waits until the previous chain's log holds its end marker
AND no matching process is running, then takes a lock file and works through its own point list. It does
NOT count itself as GPU work (the venv `python.exe` is a shim, so one chain shows two processes), and the
lock file is the only double-launch guard - delete it by hand after an outage.

Usage (detached), one line per chain:
  .venv/Scripts/python.exe -u scratchpad/quant_plan/it36ladder_chain.py \
      --name it36rankprec --prev-log it36notecut_chain.log --prev-end "=== it36notecut chain end ===" \
      --points "l2i4s4:it36l2i4s4,l4i4s4:it36l4i4s4,l7i4s4:it36l7i4s4,n2i4s4:it36n2i4s4"
A point is `candidate:stamp[:steps[:ref]]`; `steps` defaults to 124 and `ref` to `qnj_it36ctl`.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
RESULT = ROOT / "result"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ROSTER = RUN / "roster750_seed20261119.json"
START_FOLDER = "pretrain/rwkv/opt_iter36"
START_NAME = "rwkv_iter36_558"
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|it36qatsmall2_chain.py|"
            "it36qatsmall3_chain.py|it36qatsmall4_chain.py|it36rank_chain.py|it36notecut_chain.py|"
            "it36ladder_chain.py|rust_speed.py|rust_grid.py|rwkv-infer")
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


def wait_for_prev(log, prev_log, prev_end):
    t0 = time.time()
    idle = 0
    while True:
        ended = prev_end in prev_log.read_text(encoding="utf-8") if prev_log.exists() else False
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


def run(log, tag, cmd, logfile):
    log(f"START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="log and lock base name, e.g. it36rankprec")
    ap.add_argument("--prev-log", required=True, help="previous chain's log file name")
    ap.add_argument("--prev-end", required=True, help="previous chain's end marker line")
    ap.add_argument("--points", required=True,
                    help="comma-separated candidate:stamp[:steps[:ref]] entries")
    args = ap.parse_args()

    log = make_logger(RUN / f"{args.name}_chain.log")
    lock = RUN / f"{args.name}_chain.lock"
    prev_log = RUN / args.prev_log

    log(f"=== {args.name} chain start ===")
    if lock.exists():
        log(f"lock found ({lock.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {lock}")
    if not wait_for_prev(log, prev_log, args.prev_end):
        log(f"=== {args.name} chain end (without running anything) ===")
        return 1

    for entry in args.points.split(","):
        parts = entry.strip().split(":")
        cand, stamp = parts[0], parts[1]
        steps = int(parts[2]) if len(parts) > 2 and parts[2] else 124
        ref = parts[3] if len(parts) > 3 and parts[3] else "qnj_it36ctl"
        epochs = max(2, round(steps / 62))  # the decay phase keeps its own step counter
        log(f"point {cand}@{steps} -> tag {stamp}, ref {ref}, epochs {epochs}")
        rc = qat_point(log, cand, stamp, steps, ref, epochs)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    log(f"=== {args.name} chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
