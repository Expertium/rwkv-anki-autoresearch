"""Fifth screening chain: does the CARD stream need more than rank 1? (2026-11-21 07:50)

The mechanism diagnostic decided the direction. `cn8` (full-matrix int8, 8 bits per element, NO rank
truncation, 8,704/26,112 bits - 4x over the bars) scores **+0.000033 ahead / +0.000022 imm** against the
fp32 control. So an 8-bit state loses essentially nothing when it is not truncated: the rank-1 tax
(+0.000503/+0.001009 for `l1i8`) is TRUNCATION, not factor precision. A finer level (int16) cannot help.

The per-stream attribution says which stream owns it - both runs evaluated on row 4's own checkpoint:

  * `l1i8c` (card rank-1 int8, note fp32): +0.001282 ahead / +0.001142 imm
  * `l1i8n` (note rank-1 int8, card fp32): +0.000373 ahead / +0.000258 imm

The NOTE stream at rank-1 costs less than the +0.0005 ceiling on BOTH heads; the CARD stream carries the
immediate-head tax on its own. The card's 2,176-bit bar has room for more ranks, so this chain buys ranks
for the card and leaves the note at rank-1 int8:

  1. `l2i8`   card rank-2 int8 + int8 shifts -> 1,536 / 3,072 bits
  2. `l3i8`   card rank-3 int8 + int8 shifts -> 2,048 / 3,072 bits (128 bits under the card bar)
  3. `l2i8s4` card rank-2 int8 + int4 shifts -> 1,280 / 2,304 bits (int4 shifts were free at rank-1:
              row 9 `l1i8s4` scored +0.000516/+0.001030 versus row 4's +0.000503/+0.001009)
  4. `l3i8s4` card rank-3 int8 + int4 shifts -> 1,792 / 2,304 bits

Each point: a 0.04-epoch smoke, one 124-step QAT decay from `rwkv_iter36_558` (the champion's own decay
length, so the comparison stays at the same length as rows 4-7), then the quantized eval against
`qnj_it36ctl` (row 0, the identical decay with quantization OFF). Rank >= 2 has no fused CUDA kernel, so
both the decay and the eval take the Python SVD path - expect slower points than the rank-1 ladder.

Relaunch-safe, lock-guarded, does not count itself as GPU work (the venv `python.exe` is a shim, so one
chain shows two processes). Delete the lock by hand after an outage.

Usage (detached): .venv/Scripts/python.exe -u scratchpad/quant_plan/it36rank_chain.py
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
LOG = RUN / "it36rank_chain.log"
LOCK = RUN / "it36rank_chain.lock"
PREV_LOG = RUN / "it36qatsmall4_chain.log"
PREV_END = "=== it36qatsmall4 chain end ==="
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
STEPS = 124
EPOCHS = 2
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|it36qatsmall2_chain.py|"
            "it36qatsmall3_chain.py|it36qatsmall4_chain.py|rust_speed.py|rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 600
IDLE_PROCEED_MIN = 12

POINTS = [
    ("l2i8", "it36l2i8"),
    ("l3i8", "it36l3i8"),
    ("l2i8s4", "it36l2i8s4"),
    ("l3i8s4", "it36l3i8s4"),
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


def run(tag, cmd, logfile):
    log(f"START {tag}: {' '.join(cmd)}")
    t0 = time.time()
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {tag} {' '.join(cmd)} ===\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"DONE  {tag}: rc={rc} in {(time.time() - t0) / 60:.1f} min")
    return rc


def qat_point(cand, stamp):
    """0.04-epoch smoke -> one 124-step QAT decay -> the quantized eval against row 0."""
    if (RESULT / f"adw_q{cand}_{stamp}.jsonl").exists():
        log(f"{cand}@{STEPS}: result exists -- skipping the whole point")
        return 0
    base = [str(PY), "-u", str(RUN / "qat_decay.py"), "--candidate", cand,
            "--train-users", "1", "100", "--max-train-len", "66000",
            "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558"]
    rc = run(f"smoke {cand}", base + ["--epochs", "0.04", "--max-steps", "6",
                                      "--tag", f"{stamp}smoke"],
             RUN / f"{stamp}smoke_launcher.log")
    if rc != 0:
        log(f"smoke FAILED for {cand} -- skipping this point")
        return rc
    rc = run(f"qat {cand}", base + ["--epochs", str(EPOCHS), "--max-steps", str(STEPS),
                                    "--prefix", "qat", "--tag", stamp],
             RUN / f"{stamp}_launcher.log")
    if rc != 0:
        log(f"qat FAILED for {cand} -- skipping this point")
        return rc
    ckpt = RUN / f"qat_{stamp}" / f"qat_{STEPS}.pth"
    if not ckpt.exists():
        log(f"checkpoint missing: {ckpt}")
        return 1
    return run(f"eval {cand}", [str(PY), "-u", str(RUN / "quant_sweep.py"), "--model", str(ckpt),
                                "--ref", CTL_124, "--candidates", cand, "--shards", "2",
                                "--solo", "1000000", "--stamp", stamp, "--users-file", str(ROSTER)],
               RUN / f"eval_{stamp}_launcher.log")


def main():
    log("=== it36rank chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36rank chain end (without running anything) ===")
        return 1
    for cand, stamp in POINTS:
        rc = qat_point(cand, stamp)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")
    log("=== it36rank chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
