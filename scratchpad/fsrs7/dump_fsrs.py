"""Dump PER-REVIEW FSRS predictions for the decorrelation test (scratchpad/arch_2026-09-07).

WHY. Our two RWKV models -- realcyc (563k, gen-5 features) and the pretrained d=128 (2.76M,
published features) -- have a residual correlation of 0.9957 on ahead. That extends the
2026-07-03 "family saturated" result across a 5x parameter change AND a feature-layout change,
but BOTH arms are RWKV-7 with the same 5-stream structure, so it cannot separate TRUE NOISE
from a blind spot SHARED by the family. FSRS is the out-of-family contrast: a hand-designed
memory model with a completely different inductive bias.

WHAT IT RUNS. srs-benchmark's own `script.process(user_id)`, unmodified, imported from
Andrew's read-only clone. Nothing in that repo is written to: this driver chdir's into its own
output directory FIRST, so the relative paths script.py writes (`evaluation/`, `result/`) land
here. The only cost to his tree is __pycache__.

WHY `--file` AND NOT `--raw`. The raw jsonl carries only (user, p, y) -- no row key -- so it
cannot be joined to our dumps. `--file` writes the whole scored dataframe per user, including
`review_th`, which is the key our decorrelate.py already intersects on.

ALIGNMENT. FSRS reads the PUBLISHED dataset, the same one arm B (d=128) read, so review_th
means the same thing. cmp() re-checks it by requiring the labels to agree; a silent
misalignment shows up as label disagreement, not as a plausible-looking correlation.

USAGE
    dump_fsrs.py <tag> -- <srs-benchmark flags...>
    dump_fsrs.py cmp <tagA> <tagB>          compare two dumped arms
"""

import os
import sys
from pathlib import Path

SRSB = r"C:\Users\Andrew\srs-benchmark"
REPO = r"C:\Users\Andrew\rwkv-anki-autoresearch"
DATA = r"C:\Users\Andrew\anki-revlogs-10k"
USERS = [int(x) for x in os.environ.get("FSRS_USERS", "5001,5002,5003,5004").split(",")]
HERE = Path(REPO) / "scratchpad" / "fsrs7"


def dump(tag, flags):
    outdir = HERE / f"out_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(outdir)
    sys.path.insert(0, SRSB)
    sys.argv = ["script.py", "--data", DATA, "--file", "--processes", "1"] + flags
    import script  # noqa: E402  (config is parsed from sys.argv at import time)

    name = script.config.get_evaluation_file_name()
    print(f"[{tag}] eval name = {name}   device = {script.config.device}", flush=True)
    Path("evaluation", name).mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(exist_ok=True)
    for uid in USERS:
        res, err = script.process(uid)
        if err is not None:
            print(f"[{tag}] user {uid} FAILED:\n{err}", flush=True)
            sys.exit(3)
        stats = res[0]
        keep = ("size", "LogLoss", "logloss", "RMSE(bins)", "RMSE")
        print(f"[{tag}] user {uid}: "
              + " ".join(f"{k}={stats[k]}" for k in keep if k in stats), flush=True)
        tsv = Path("evaluation", name, f"{uid}.tsv")
        if not tsv.exists():
            print(f"[{tag}] user {uid}: NO TSV at {tsv} -- --file did not take", flush=True)
            sys.exit(4)
    # Collapse the TSVs into one compact npz keyed by (user, review_th).
    import numpy as np
    import pandas as pd

    frames = []
    for uid in USERS:
        d = pd.read_csv(Path("evaluation", name, f"{uid}.tsv"), sep="\t",
                        usecols=["review_th", "y", "p"])
        d["u"] = uid
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    assert not df.duplicated(["u", "review_th"]).any(), "a review is scored twice"
    np.savez_compressed(HERE / f"fsrs_{tag}.npz", u=df["u"].to_numpy(np.int32),
                        rth=df["review_th"].to_numpy(np.int64),
                        y=df["y"].to_numpy(np.int8), p=df["p"].to_numpy(np.float64))
    print(f"[{tag}] wrote fsrs_{tag}.npz  rows={len(df)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "cmp":
        raise SystemExit("use cmp_fsrs.py")
    tag = sys.argv[1]
    sep = sys.argv.index("--")
    dump(tag, sys.argv[sep + 1:])
