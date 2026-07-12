"""Live champion-vs-candidate training-loss viewer (matplotlib window, refreshes every 15 s).

Auto-discovers the ACTIVE run: newest *_ws_trace.jsonl under scratchpad/tuner5k/*/ or
scratchpad/*/ (HP-tuner trials and champion runs both write RWKV_STEP_TRACE), excluding the
current champion's own trace. Champion reference = optimization/champion_5k.json (embedded
trace). Two panels (ahead / imm): EMA-smoothed curves for display; the p-value is a paired
one-sided Wilcoxon (candidate better) on the RAW per-step diffs over the common window --
the same pairing the tuner's prune test uses. Vertical lines: warmup end + WS end (= decay
phase start; the trace itself covers the WS phase). When a new trial starts, the plot
switches to it automatically on the next refresh.

Usage:
  python scratchpad/liveplot/liveplot.py            # live window, 15 s refresh
  python scratchpad/liveplot/liveplot.py --once     # render one frame to liveplot_test.png
Close the window (or Ctrl+C) to stop.
"""
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np

ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"
os.chdir(ROOT)
ONCE = "--once" in sys.argv
import matplotlib
matplotlib.use("Agg" if ONCE else "TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

REFRESH_S = 15
C_CHAMP = "#2196F3"   # blue (like the sketch)
C_CAND = "#23A455"    # green
C_INK = "#333333"
C_LINEMARK = "#888888"


def load_champion():
    with open("optimization/champion_5k.json") as fh:
        d = json.load(fh)
    # step-indexed dense arrays (trace_step is 1..N)
    return {
        "name": d["name"],
        "ahead": np.asarray(d["trace_ahead"], dtype=np.float64),
        "imm": np.asarray(d["trace_imm"], dtype=np.float64),
        "n": int(d["n_trace_steps"]),
    }


def discover_trace(champ_name):
    cands = glob.glob("scratchpad/tuner5k/*/*_ws_trace.jsonl") + glob.glob("scratchpad/*/*_ws_trace.jsonl")
    cands = [p for p in cands if champ_name not in os.path.basename(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def read_trace(path):
    steps, ahead, imm = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial trailing line mid-write
            steps.append(r["step"])
            ahead.append(r["ahead"])
            imm.append(r["imm"])
    steps = np.asarray(steps)
    ahead = np.asarray(ahead, dtype=np.float64)
    imm = np.asarray(imm, dtype=np.float64)
    # STEP_TRACE appends: a re-run trial (config revisited across eras) leaves the old run's
    # lines in front. Keep only the last monotonic segment (after the final step reset).
    resets = np.where(np.diff(steps) < 0)[0]
    if len(resets):
        cut = resets[-1] + 1
        steps, ahead, imm = steps[cut:], ahead[cut:], imm[cut:]
    return steps, ahead, imm


def run_meta(trace_path):
    """(run_name, warmup_steps, ws_steps) from the trial sidecar json, else the WS toml."""
    folder = os.path.dirname(trace_path)
    name = os.path.basename(trace_path).replace("_ws_trace.jsonl", "")
    warmup, ws_steps = 200, None
    sidecar = os.path.join(folder, f"{name}.json")
    if os.path.exists(sidecar):
        try:
            sc = json.load(open(sidecar))
            warmup = int(sc["config"]["warmup_steps"])
            ws_steps = int(sc.get("ws_steps") or 0) or None
        except Exception:
            pass
    else:
        for toml in glob.glob(os.path.join(folder, "*_ws.toml")) + glob.glob(os.path.join(folder, "*ws.toml")):
            m = re.search(r"^WARMUP_STEPS\s*=\s*(\d+)", open(toml).read(), re.M)
            if m:
                warmup = int(m.group(1))
                break
    return name, warmup, ws_steps


def ema(x, span):
    if len(x) == 0:
        return x
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


DELTA_TAIL = 100   # mean delta over the last N paired steps (full-window mean is inflated
                   # by the early transient where both losses are huge and far apart)
P_WINDOW = 1500    # Wilcoxon over the last N paired steps -- MATCHES the pruner's
                   # RWKV_PRUNE_WINDOW (2026-07-08: full-window drags stale early history)


def paired_p(champ_dense, cand_steps, cand_vals):
    """One-sided Wilcoxon p that the candidate is BETTER (lower loss), paired by step,
    over the last P_WINDOW paired steps (same window the pruner tests). Delta = last-N mean
    of (cand - champ): NEGATIVE = candidate better (lower loss), sign matches the metric."""
    ok = cand_steps <= len(champ_dense)
    if ok.sum() < 20:
        return None, None
    d = cand_vals[ok] - champ_dense[cand_steps[ok] - 1]  # <0 = candidate better
    d_tail = float(d[-DELTA_TAIL:].mean())
    nz = d[-P_WINDOW:]
    nz = nz[nz != 0]
    if len(nz) < 20:
        return None, d_tail
    return float(wilcoxon(nz, alternative="less").pvalue), d_tail


def draw(fig, axes, champ, trace_path):
    steps, cand_a, cand_i = read_trace(trace_path)
    name, warmup, ws_steps = run_meta(trace_path)
    ws_steps = ws_steps or champ["n"]
    span = max(25, len(steps) // 30)
    # NOTE: the trace's "imm" is the 4-WAY rating cross-entropy (chance = log4 = 1.39), NOT the
    # benchmark's binary imm logloss -- so it sits above "ahead" (binary curve BCE, chance = log2)
    # even though benchmark-imm < benchmark-ahead. Same fields in both traces -> pairing is valid.
    panels = ((axes[0], "ahead", "ahead loss (curve BCE, binary)", champ["ahead"], cand_a),
              (axes[1], "imm", "imm loss (4-way rating CE)", champ["imm"], cand_i))
    for ax, mode, ylab, champ_dense, cand_v in panels:
        ax.clear()
        n_show = min(len(steps), champ["n"]) if len(steps) else champ["n"]
        cx = np.arange(1, max(n_show, 2) + 1)
        ax.plot(cx, ema(champ_dense[: len(cx)], span), color=C_CHAMP, lw=2.2,
                label=f"champion ({champ['name']})")
        if len(steps):
            ax.plot(steps, ema(cand_v, span), color=C_CAND, lw=1.6, label=f"candidate ({name})")
        p, dmean = paired_p(champ_dense, steps, cand_v) if len(steps) else (None, None)
        ptxt = (f"p(cand better, last {P_WINDOW}) = n/a" if p is None
                else f"p(cand better, last {P_WINDOW}) = {p:.2g}")
        if dmean is not None:
            ptxt += f"\ndelta (last {DELTA_TAIL} steps) = {dmean:+.4f} (neg = cand better)"
        ax.text(0.985, 0.945, ptxt, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=C_INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85))
        if warmup and warmup < (steps[-1] if len(steps) else champ["n"]) * 1.05:
            ax.axvline(warmup, color=C_LINEMARK, lw=1.0, ls="--", alpha=0.8)
            ax.text(warmup, 0.02, " warmup end", transform=ax.get_xaxis_transform(),
                    fontsize=8, color=C_LINEMARK, ha="left", va="bottom")
        ax.axvline(ws_steps, color=C_LINEMARK, lw=1.0, ls=":", alpha=0.8)
        ax.text(ws_steps, 0.02, "decay starts ", transform=ax.get_xaxis_transform(),
                fontsize=8, color=C_LINEMARK, ha="right", va="bottom")
        # y-window: ignore the first few % of steps (huge init losses squash the tail)
        ref = []
        skip = max(10, int(0.03 * len(cx)))
        ref.append(ema(champ_dense[: len(cx)], span)[skip:])
        if len(steps) > skip:
            ref.append(ema(cand_v, span)[skip:])
        allv = np.concatenate(ref) if ref else np.array([0, 1])
        pad = 0.06 * (allv.max() - allv.min() + 1e-9)
        ax.set_ylim(allv.min() - pad, allv.max() + 3 * pad)
        ax.set_xlim(0, ws_steps * 1.02)
        ax.set_ylabel(ylab, color=C_INK)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    axes[1].set_xlabel("WS step", color=C_INK)
    last = steps[-1] if len(steps) else 0
    fig.suptitle(f"{name}  vs  champion — WS step {last:,}/{ws_steps:,}   "
                 f"(updated {time.strftime('%H:%M:%S')}, refresh {REFRESH_S}s)",
                 fontsize=11, color=C_INK)


def main():
    champ = load_champion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if not ONCE:
        fig.canvas.manager.set_window_title("RWKV 5k: champion vs candidate")
        plt.show(block=False)  # map the window ONCE; refreshes must never call show/pause
                               # again (plt.pause re-raises the window = steals focus)
    did_layout = False
    while True:
        # NOTHING may kill the window: a toolbar save can raise from deep inside the
        # renderer (seen live: FT_Render_Glyph "raster overflow"), and those exceptions
        # surface here via flush_events/draw_idle. Log + continue, always.
        try:
            trace = discover_trace(champ["name"])
            if trace:
                try:
                    draw(fig, axes, champ, trace)
                except Exception as e:  # file mid-rotation etc. -- keep the window alive
                    axes[0].set_title(f"(draw error, retrying: {e})", fontsize=9)
            else:
                axes[0].clear()
                axes[0].set_title("no candidate trace found yet -- waiting")
            if not did_layout:
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                did_layout = True
            if ONCE:
                out = os.path.join("scratchpad", "liveplot", "liveplot_test.png")
                fig.savefig(out, dpi=110)
                print(f"saved {out}")
                return
            fig.canvas.draw_idle()
        except Exception:
            print(f"[{time.strftime('%H:%M:%S')}] refresh-cycle error (window kept alive):")
            traceback.print_exc()
        # stay responsive between refreshes; exit when the window is closed.
        # flush_events (NOT plt.pause) -- pause calls show() every tick, stealing focus.
        t0 = time.time()
        while time.time() - t0 < REFRESH_S:
            if not plt.fignum_exists(fig.number):
                return
            try:
                fig.canvas.flush_events()
            except Exception:
                print(f"[{time.strftime('%H:%M:%S')}] flush_events error (window kept alive):")
                traceback.print_exc()
            time.sleep(0.25)


if __name__ == "__main__":
    main()
