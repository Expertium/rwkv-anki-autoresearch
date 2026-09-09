"""Plot the endgame run's train and validation loss: ws10's WS phase + a decay branch.

Andrew 2026-09-10: "Plot the train and val loss of the current run (with the number of users
displayed)." The user counts are on every panel because they are what makes the curves readable:
TRAIN is 5,000 users and VALIDATION is TEN (5001-5010, 594,215 rows), so the validation line is a
low-variance estimate of a narrow slice -- not a small sample of the 2,499-user gate set.

WHAT IS AND IS NOT COMPARABLE -- the rule gap_trace.py carries, enforced here by giving the
incomparable pair separate panels instead of a shared axis:
  * AHEAD: train and validation are both the ahead BCE, so they share an axis. The LEVEL still
    carries a constant offset (train is logged with dropout ON, at the stable LR, on the train
    database), so read the TRAJECTORY and the movement of the gap, never the raw distance.
  * IMM: the training number is the 4-way CROSS-ENTROPY of the rating head; the validation number
    is the BINARY logloss of 1-P(Again). Different quantities -- plotting them together would
    manufacture a ~-0.24 "gap" that is a units mismatch. One panel each.

Y-AXES ARE CLIPPED to the post-warm-up range. Step 50 sits at ahead 0.61 / imm 0.39 and would
compress every later movement into a flat line; the clipped range is stated on the panel.

Usage: python scratchpad/ws10/plot_losses.py [out.png]
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN = re.compile(r"^(\d+) (\d+) (\d+), all: ([\d.]+), ahead: ([\d.]+) \([\d.]+\), imm: ([\d.]+)")
VAL = re.compile(
    r"^Mean ahead validation loss: ([\d.]+) \([\d.]+\), imm: ([\d.]+), validation n: (\d+)"
)

WS_LOG = "scratchpad/ws10/ws_*.log"
DECAY_LOG = "scratchpad/w10plain/decay_*.log"
REF_LOG = "scratchpad/realcyc/decay_*.log"   # realcyc = the 1.25-epoch reference, SAME 10 val users
WS_STEPS = 109350          # ws10: 10 epochs x 10,935 groups
GROUPS = 10935             # optimizer steps per epoch
SMOOTH = 400               # train steps per rolling mean; the per-step loss is very noisy
ZOOM_SMOOTH = 100          # finer window for the decay-only panel
WARM = 2000                # steps ignored when choosing the y-range

INK, INK2, MUTED = "#1b1b1b", "#5a5a5a", "#9a9a9a"
C_TRAIN, C_VAL = "#4a7ebb", "#c0504d"


def parse(pattern, offset):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not paths:
        sys.exit("no log matching " + pattern)
    with open(paths[-1], encoding="utf-8", errors="replace") as fh:
        text = fh.read().replace("\r", "\n")
    tr, va = [], []
    last = None
    for line in text.split("\n"):
        m = TRAIN.match(line)
        if m:
            last = int(m.group(3)) + offset
            tr.append((last, float(m.group(5)), float(m.group(6))))
            continue
        m = VAL.match(line)
        if m and last is not None:
            va.append((last, float(m.group(1)), float(m.group(2)), int(m.group(3))))
    return os.path.basename(paths[-1]), tr, va


def roll(rows, idx, window, lo=None):
    """Rolling mean over `window` train steps, emitted once per window (no overlap)."""
    xs, ys, acc = [], [], []
    for r in rows:
        if lo is not None and r[0] < lo:
            continue
        acc.append(r[idx])
        if len(acc) == window:
            xs.append(r[0])
            ys.append(sum(acc) / window)
            acc = []
    return xs, ys


def frame(ax, title, sub, vline=True, pad=16, sub_y=1.012):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    if vline:
        ax.axvline(WS_STEPS, color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=pad)
    ax.text(0.0, sub_y, sub, transform=ax.transAxes, fontsize=8.5, color=INK2)


def span(series, pad=0.06):
    lo = min(min(s) for s in series if s)
    hi = max(max(s) for s in series if s)
    m = (hi - lo) * pad
    return lo - m, hi + m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/ws10/loss_curves.png"
    ws_name, ws_tr, ws_va = parse(WS_LOG, 0)
    d_name, d_tr, d_va = parse(DECAY_LOG, WS_STEPS)
    try:
        r_name, _, r_va = parse(REF_LOG, 0)
    except SystemExit:
        r_name, r_va = None, []
    ref_a = r_va[-1][1] if r_va else None
    ref_i = r_va[-1][2] if r_va else None

    val_n = ws_va[-1][3] if ws_va else 0
    end = d_tr[-1][0] if d_tr else WS_STEPS

    ax_x, ax_y = roll(ws_tr, 1, SMOOTH)
    dax_x, dax_y = roll(d_tr, 1, SMOOTH)
    ai_x, ai_y = roll(ws_tr, 2, SMOOTH)
    dai_x, dai_y = roll(d_tr, 2, SMOOTH)
    zx, zy = roll(d_tr, 1, ZOOM_SMOOTH)

    va_x = [r[0] for r in ws_va] + [r[0] for r in d_va]
    va_a = [r[1] for r in ws_va] + [r[1] for r in d_va]
    va_i = [r[2] for r in ws_va] + [r[2] for r in d_va]
    lateA = [(x, y) for x, y in zip(va_x, va_a) if x >= WARM]
    lateI = [(x, y) for x, y in zip(va_x, va_i) if x >= WARM]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2))
    fig.suptitle("ws10 (10 epochs WS) + endgame arm 1 (2 epochs decay) -- train vs validation",
                 color=INK, fontsize=14, x=0.012, ha="left", y=0.988)

    # ---- (0,0) ahead over the whole run ----------------------------------------------------
    ax = axes[0][0]
    frame(ax, "ahead BCE -- the one comparable pair",
          "train 5,000 users (dropout on)  vs  validation 10 users, {:,} rows (dropout off)"
          .format(val_n), pad=46, sub_y=1.135)
    ax.plot(ax_x, ax_y, color=C_TRAIN, linewidth=1.4,
            label="train, {}-step mean  (5,000 users)".format(SMOOTH))
    ax.plot(dax_x, dax_y, color=C_TRAIN, linewidth=1.4)
    ax.plot(va_x, va_a, color=C_VAL, linewidth=1.8, marker="o", markersize=2.6,
            label="validation  (10 users)")
    lo, hi = span([[y for x, y in lateA], [y for x, y in zip(ax_x, ax_y) if x >= WARM]])
    ax.set_ylim(lo, hi)
    ax.set_ylabel("ahead loss", color=INK, fontsize=10)
    ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.text(WS_STEPS - 1500, lo + (hi - lo) * 0.04, "WS ends 109,350  |  decay",
            fontsize=8.5, color=INK2, ha="right")
    ax.text(0.02, 0.06, "step 50 is off scale at 0.614", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    top = ax.secondary_xaxis("top", functions=(lambda s: s / GROUPS, lambda e: e * GROUPS))
    top.set_xlabel("epochs", color=MUTED, fontsize=8)
    top.tick_params(colors=MUTED, labelsize=8)

    # ---- (0,1) the decay phase alone -- the branch under test ------------------------------
    ax = axes[0][1]
    frame(ax, "ahead BCE, DECAY PHASE ONLY -- what arm 1 buys",
          "the 2-epoch cosine decay off w10_ws_109350 -- validation moves 0.3209 -> 0.3210 "
          "(10 users)", vline=False)
    ax.plot(zx, zy, color=C_TRAIN, linewidth=1.4,
            label="train, {}-step mean  (5,000 users)".format(ZOOM_SMOOTH))
    dv_x = [r[0] for r in d_va]
    dv_y = [r[1] for r in d_va]
    ax.plot(dv_x, dv_y, color=C_VAL, linewidth=1.8, marker="o", markersize=4.0,
            label="validation  (10 users)")
    ax.axhline(ws_va[-1][1], color=MUTED, linewidth=1.0, linestyle=":")
    if ref_a is not None:
        ax.axhline(ref_a, color="#7f5aa8", linewidth=1.4, linestyle="--",
                   label="realcyc, 1.25 epochs, same 10 users: {:.4f}".format(ref_a))
    ax.set_ylabel("ahead loss", color=INK, fontsize=10)
    ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.annotate("final {:.4f}".format(dv_y[-1]), xy=(dv_x[-1], dv_y[-1]),
                xytext=(-96, -30), textcoords="offset points", fontsize=9, color=C_VAL,
                arrowprops=dict(arrowstyle="-", color=C_VAL, linewidth=0.9))

    # ---- (1,0) imm TRAIN -------------------------------------------------------------------
    ax = axes[1][0]
    frame(ax, "imm TRAIN -- 4-way cross-entropy",
          "rating head over Again/Hard/Good/Easy, 5,000 users -- NOT the panel beside it")
    ax.plot(ai_x, ai_y, color=C_TRAIN, linewidth=1.4)
    ax.plot(dai_x, dai_y, color=C_TRAIN, linewidth=1.4)
    lo, hi = span([[y for x, y in zip(ai_x, ai_y) if x >= WARM], dai_y])
    ax.set_ylim(lo, hi)
    ax.set_ylabel("4-way CE", color=INK, fontsize=10)
    ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax.text(0.02, 0.06, "step 50 is off scale at 1.006", transform=ax.transAxes,
            fontsize=8, color=MUTED)

    # ---- (1,1) imm VAL ---------------------------------------------------------------------
    ax = axes[1][1]
    frame(ax, "imm VALIDATION -- binary logloss of 1-P(Again)",
          "10 users, {:,} rows -- a DIFFERENT quantity from the panel beside it".format(val_n))
    ax.plot(va_x, va_i, color=C_VAL, linewidth=1.8, marker="o", markersize=2.6)
    if ref_i is not None:
        ax.axhline(ref_i, color="#7f5aa8", linewidth=1.4, linestyle="--")
        ax.text(1000, ref_i, " realcyc, 1.25 epochs: {:.4f}".format(ref_i), fontsize=8.5,
                color="#7f5aa8", va="bottom")
    lo, hi = span([[y for x, y in lateI]])
    ax.set_ylim(lo - (hi - lo) * 0.10, hi)
    ax.set_ylabel("binary logloss", color=INK, fontsize=10)
    ax.set_xlabel("optimizer step", color=INK2, fontsize=9)
    ax.text(0.02, 0.06, "step 50 is off scale at 0.392", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.annotate("final {:.4f}".format(va_i[-1]), xy=(va_x[-1], va_i[-1]),
                xytext=(-104, -24), textcoords="offset points", fontsize=9, color=C_VAL,
                arrowprops=dict(arrowstyle="-", color=C_VAL, linewidth=0.9))

    fig.text(0.012, 0.012,
             "train users 1-5000 (5,000)   |   validation users 5001-5010 (10 users, {:,} rows, "
             "held out, dropout off)   |   the graded VAL-half eval is 2,499 users and is still "
             "running\nlogs: {} + {}   |   gen-5 dbs, KD off, augmentation off, "
             "{:,} optimizer steps total".format(val_n, ws_name, d_name, end),
             fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.062, right=0.988, top=0.865, bottom=0.095, hspace=0.46,
                        wspace=0.17)
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote " + out)
    print("WS train {:,} pts / {} vals; decay train {:,} pts / {} vals".format(
        len(ws_tr), len(ws_va), len(d_tr), len(d_va)))
    print("val ahead: step50 {:.4f}  1ep {:.4f}  WS-end {:.4f}  decay-end {:.4f}".format(
        va_a[0], [y for x, y in zip(va_x, va_a) if x >= GROUPS][0], ws_va[-1][1], va_a[-1]))
    if ref_a is not None:
        print("realcyc (1.25 ep) final val: ahead {:.4f} imm {:.4f}  -> 10+2 moves "
              "ahead {:+.4f}, imm {:+.4f}".format(ref_a, ref_i, va_a[-1] - ref_a,
                                                  va_i[-1] - ref_i))
    print("val imm  : step50 {:.4f}  1ep {:.4f}  WS-end {:.4f}  decay-end {:.4f}".format(
        va_i[0], [y for x, y in zip(va_x, va_i) if x >= GROUPS][0], ws_va[-1][2], va_i[-1]))


if __name__ == "__main__":
    main()
