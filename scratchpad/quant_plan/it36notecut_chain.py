"""Sixth screening chain: can the NOTE pool be cut? Plus the card-rank x note-cut combinations.
(2026-11-21 08:05)

The note stream holds 2/3 of the bits (3,072 of row 4's 4,096) and its OWN rank-1 int8 tax is small
(+0.000373 ahead / +0.000258 imm - inside the ceiling by itself), while the card carries the immediate
head tax (+0.001142). Two facts therefore pull in opposite directions: the note is where the bits are,
but the card is where the tax is. This chain measures the note's int4 tax, then (conditionally) combines
the note cut with the card rank fix that `it36rank` is measuring.

Points:
  1. `nr4`   note rank-1 int4 + int4 shifts, card at row 4's EXACT config -> 1,024 / 1,536 bits
  2. `nr4s2` note rank-1 int4 + int2 shifts, same card -> 1,024 / 1,152 bits
  3. `l2i8s4n4` card rank-2 int8 + int4 shifts, note rank-1 int4 + int4 -> 1,280 / 1,536 bits
  4. `l3i8s4n4` card rank-3 int8 + int4 shifts, note rank-1 int4 + int4 -> 1,792 / 1,536 bits

Points 3-4 are gated on the rank ladder actually helping: they only run if the best card-rank point in
`it36rank` came in under +0.00070 imm (clearly better than row 4's +0.001009). Otherwise the combination
cannot pass and the two runs are wasted.

Everything else is as in `it36rank_chain.py`: 0.04-epoch smoke, one 124-step QAT decay from
`rwkv_iter36_558` (the champion's own decay length), quantized eval against `qnj_it36ctl` (row 0).
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
LOG = RUN / "it36notecut_chain.log"
LOCK = RUN / "it36notecut_chain.lock"
PREV_LOG = RUN / "it36rank_chain.log"
PREV_END = "=== it36rank chain end ==="
ROSTER = RUN / "roster750_seed20261119.json"
CTL_124 = "qnj_it36ctl"
STEPS = 124
EPOCHS = 2
RANK_STAMPS = ("it36l2i8", "it36l3i8", "it36l2i8s4", "it36l3i8s4")
RANK_IMM_GATE = 0.00070   # proceed with the combinations only below this imm delta
BUSY_PAT = ("quant_sweep.py|rwkv.train_rwkv|it36qatsmall_chain.py|it36qatsmall2_chain.py|"
            "it36qatsmall3_chain.py|it36qatsmall4_chain.py|it36rank_chain.py|rust_speed.py|"
            "rust_grid.py|rwkv-infer")
WAIT_MAX_MIN = 600
IDLE_PROCEED_MIN = 12

POINTS = [("nr4", "it36nr4"), ("nr4s2", "it36nr4s2")]
COMBO_POINTS = [("l2i8s4n4", "it36l2i8s4n4"), ("l3i8s4n4", "it36l3i8s4n4")]


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


def best_rank_imm():
    """The best (smallest) imm delta the rank ladder produced, or None if it is not all there."""
    best = None
    for s in RANK_STAMPS:
        p = RUN / f"quant_sweep_{s}.json"
        if not p.exists():
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
            d = rows[0]["gate"]["imm"]["delta"]
        except (KeyError, IndexError, ValueError) as exc:
            log(f"could not read the imm delta from {p.name}: {exc}")
            continue
        log(f"rank ladder: {s} imm delta {d:+.6f}")
        best = d if best is None else min(best, d)
    return best


def main():
    log("=== it36notecut chain start ===")
    if LOCK.exists():
        log(f"lock found ({LOCK.read_text(encoding='utf-8').strip()}) -- another copy is going; exiting")
        return 1
    LOCK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log(f"lock taken: {LOCK}")
    if not wait_for_prev_chain():
        log("=== it36notecut chain end (without running anything) ===")
        return 1

    for cand, stamp in POINTS:
        rc = qat_point(cand, stamp)
        if rc != 0:
            log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    best = best_rank_imm()
    if best is None:
        log("combinations: no rank-ladder result to read -- skipping the card-rank x note-cut points")
    elif best > RANK_IMM_GATE:
        log(f"combinations: the best card-rank imm delta {best:+.6f} is above {RANK_IMM_GATE:+.5f}, so a "
            f"card-rank x note-cut combination cannot pass -- skipping those two points")
    else:
        log(f"combinations: the best card-rank imm delta {best:+.6f} is below {RANK_IMM_GATE:+.5f} -- "
            f"running the combinations")
        for cand, stamp in COMBO_POINTS:
            rc = qat_point(cand, stamp)
            if rc != 0:
                log(f"{cand}: point returned rc={rc} -- continuing with the next point")

    log("=== it36notecut chain end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
