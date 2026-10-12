"""Exit 1 if a training log skipped more than MAX_FRAC of STEPS batches ("Exception caught ... Skipping batch").
A VRAM cap (RWKV_VRAM_CAP_GB) turns an over-cap allocation into a skipped batch, so a capped run must count them.
Optional --max-val-fail N (2026-09-29): also exit 1 if more than N validations failed ("Exception in validate"),
for runs whose validation trace is the instrument (WS trials: the kill rule and the endpoint ranking).
Usage: count_skips.py <log> <steps> [max_frac=0.01] [--max-val-fail N]"""
import io, sys
args = [a for a in sys.argv[1:]]
mvf = None
if "--max-val-fail" in args:
    i = args.index("--max-val-fail"); mvf = int(args[i + 1]); del args[i:i + 2]
n = vf = 0
for l in io.open(args[0], encoding="utf-8", errors="replace"):
    if l.startswith("Exception caught"):
        n += 1
    elif l.startswith("Exception in validate"):
        vf += 1
steps = int(args[1]); frac = float(args[2]) if len(args) > 2 else 0.01
print("skipped batches: %d of %d steps (limit %.1f%%); failed validations: %d%s"
      % (n, steps, 100 * frac, vf, "" if mvf is None else " (limit %d)" % mvf))
sys.exit(1 if n > frac * steps or (mvf is not None and vf > mvf) else 0)
