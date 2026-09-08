"""Is a GENERALISATION GAP opening at the 10-epoch budget? Read it off ws10's own log, live.

WHY THIS DECIDES THE DECAY-BRANCH QUEUE. iter 65 (SAM) regressed at 1.25 epochs and the train loss
rose WITH the held-out loss, which is the signature of a FIT-limited model: there is no gap for a
regulariser to close. Four readings agreed, and the family was deprioritised with an explicit
condition attached -- "re-screen them on the 10x endgame's checkpoints, where a gap CAN exist".
ws10 validates every 1,000 steps over 594,215 held-out rows, so that screen is being produced for
free right now and needs no GPU of its own.

  gap FLAT or shrinking  -> the model is still fit-limited at 10 epochs; SAM / wd / dropout stay
                            unmotivated and the decay branches should go elsewhere.
  gap OPENING            -> the condition is met; regularisation becomes the priority branch set,
                            and it is affordable now that a branch is decay-only.

WHAT IS AND IS NOT COMPARABLE.
* AHEAD is comparable: both numbers are the ahead BCE. Train is logged WITH dropout on at the
  stable LR and validation with dropout off on another database, so the LEVEL carries a constant
  offset; the TRAJECTORY is the signal, because that offset does not move across the run.
* IMM IS NOT COMPARABLE AND ITS COLUMN IS DIAGNOSTIC ONLY. The training `imm_loss` is the 4-way
  cross-entropy of the RATING head; the validation `imm` is the binary logloss of 1-P(Again).
  Different quantities, which is the whole of the ~-0.24 "gap" -- it is a units mismatch, not
  generalisation. Printed because its trajectory is still worth a glance; never verdicted.

Reads the newest scratchpad/ws10/ws_*.log. Train steps are `<epoch> <step_in_epoch> <global>, all:
..., ahead: X (X), imm: Y`; validations are `Mean ahead validation loss: A (A), imm: B, ...`, and a
validation's global step is taken from the last train line before it.
"""

import glob
import os
import re
import sys

TRAIN = re.compile(r"^(\d+) (\d+) (\d+), all: ([\d.]+), ahead: ([\d.]+) \([\d.]+\), imm: ([\d.]+)")
VAL = re.compile(r"^Mean ahead validation loss: ([\d.]+) \([\d.]+\), imm: ([\d.]+)")
WINDOW = 400  # train steps averaged just before each validation; the per-step loss is very noisy


def main():
    logs = sorted(glob.glob("scratchpad/ws10/ws_*.log"), key=os.path.getmtime)
    if not logs:
        sys.exit("no ws10 training log")
    path = logs[-1]
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read().replace("\r", "\n")

    recent_a, recent_i, steps = [], [], []
    rows = []
    for line in text.split("\n"):
        m = TRAIN.match(line)
        if m:
            steps.append(int(m.group(3)))
            recent_a.append(float(m.group(5)))
            recent_i.append(float(m.group(6)))
            if len(recent_a) > WINDOW:
                recent_a.pop(0); recent_i.pop(0); steps.pop(0)
            continue
        m = VAL.match(line)
        if m and len(recent_a) >= WINDOW // 2:
            rows.append((steps[-1], sum(recent_a) / len(recent_a), float(m.group(1)),
                         sum(recent_i) / len(recent_i), float(m.group(2))))

    if not rows:
        sys.exit("no paired train/validation points yet")
    print(f"log {os.path.basename(path)}   {len(rows)} paired points   "
          f"train mean over the {WINDOW} steps before each validation\n")
    print(f"{'step':>8}{'epoch':>7}{'tr ahead':>10}{'val ahead':>11}{'gap':>9}"
          f"{'tr imm*':>9}{'val imm*':>10}")
    for st, ta, va, ti, vi in rows:
        print(f"{st:>8,}{st / 10935.0:>7.2f}{ta:>10.4f}{va:>11.4f}{va - ta:>+9.4f}"
              f"{ti:>9.4f}{vi:>10.4f}")

    # The verdict is about the TRAJECTORY, so compare the first and last thirds of the points
    # that are past warm-up (the first epoch moves too fast to be part of a trend).
    late = [r for r in rows if r[0] > 10935]
    if len(late) >= 6:
        n = len(late) // 3
        first, last = late[:n], late[-n:]
        for name, ai, vi_ in (("ahead", 1, 2),):  # imm units differ; see the header
            g0 = sum(r[vi_] - r[ai] for r in first) / n
            g1 = sum(r[vi_] - r[ai] for r in last) / n
            v0 = sum(r[vi_] for r in first) / n
            v1 = sum(r[vi_] for r in last) / n
            print(f"\n{name}: gap {g0:+.4f} -> {g1:+.4f} ({g1 - g0:+.4f})   "
                  f"val {v0:.4f} -> {v1:.4f} ({v1 - v0:+.4f})")
            if g1 - g0 > 0.002 and v1 > v0:
                print("  => OPENING and validation is worse: regularisation is now motivated")
            elif g1 - g0 > 0.002:
                print("  => gap opening but validation still improving: watch, do not act yet")
            else:
                print("  => still FIT-limited: regularisation stays unmotivated at this budget")


if __name__ == "__main__":
    main()
