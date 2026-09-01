"""Does rwkv/fsrs_core.py reproduce srs-benchmark's FSRS7 exactly?

A PORT THAT IS NOT CHECKED AGAINST ITS SOURCE IS A GUESS. Writing rwkv/fsrs_core.py from a
reading of models/fsrs_v7.py already produced three real errors that only appeared when the
source was read line by line rather than recalled:
  1. init_s_short is 0.8 * init_s_long, not init_s_long;
  2. the post-lapse short-term reset (cap s_short at 0.8 * post-lapse long-term S on rating==1)
     was missing entirely;
  3. s_min / D_MIN / D_MAX clamps were applied on outputs only, not on the incoming state.
None of the three would raise. All three would train, and produce a model that is "FSRS-like"
and not FSRS. Hence this file.

HOW IT RUNS. srs-benchmark is READ-ONLY to this repo and has its own venv, so the comparison
runs UNDER THAT VENV and loads our module by absolute path (fsrs_core imports nothing but
torch, which is what makes that possible). Both sides then execute in one process on identical
inputs, which is the only way a numeric comparison means anything.

WHAT IS COMPARED. Per review step, on real (delta_t, rating) sequences: the retrievability the
model predicts, and all three state components afterwards. Tolerance is 1e-5 relative -- this is
a formula port in the same dtype, so it should agree to float32 noise, not merely "closely".

Usage (from the repo root, with srs-benchmark's python):
    C:/Users/Andrew/srs-benchmark/.venv/Scripts/python.exe scratchpad/hybrid100k/smoke_fsrs_port.py
"""
import importlib.util
import os
import sys

SRS = r"C:\Users\Andrew\srs-benchmark"
OURS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rwkv",
                    "fsrs_core.py")

sys.path.insert(0, SRS)
os.chdir(SRS)

import numpy as np   # noqa: E402
import torch         # noqa: E402

torch.set_num_threads(1)
from config import Config, create_parser   # noqa: E402
from models.fsrs_v7 import FSRS7           # noqa: E402

spec = importlib.util.spec_from_file_location("fsrs_core", os.path.abspath(OURS))
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def make_config():
    parser = create_parser()
    args = parser.parse_args(["--algo", "FSRS-7", "--short", "--secs",
                              "--equalize_test_with_non_secs", "--processes", "1"])
    cfg = Config(args)
    cfg.device = torch.device("cpu")
    return cfg


cfg = make_config()
model = FSRS7(cfg).to(cfg.device).double()
w_ref = model.w.detach().double()
print("srs-benchmark FSRS7 built; %d parameters, s_min=%g" % (w_ref.numel(), cfg.s_min))
assert w_ref.numel() == fc.N_PARAMS, "parameter count disagrees"
assert abs(cfg.s_min - fc.S_MIN) < 1e-12, (
    "S_MIN mismatch: ours %g vs benchmark %g" % (fc.S_MIN, cfg.s_min))

# ---- inputs: a spread of realistic sequences, including the awkward cases ----------------
rng = np.random.default_rng(0)
B, T = 64, 24
delta_t = np.concatenate([
    rng.choice([0.0, 0.003, 0.02], size=(B, T // 3)),          # same-day (secs-scale) gaps
    rng.uniform(1.0, 30.0, size=(B, T // 3)),                  # ordinary review gaps
    rng.uniform(30.0, 3000.0, size=(B, T - 2 * (T // 3))),     # long lapses
], axis=1)
rng.shuffle(delta_t, axis=1)
rating = rng.integers(1, 5, size=(B, T)).astype(np.float64)
# force some lapses and some Easy presses into every column so the rating branches are covered
rating[: B // 4, :] = 1.0
rating[B // 4: B // 2, :] = 4.0

dt_t = torch.tensor(delta_t, dtype=torch.float64)
rt_t = torch.tensor(rating, dtype=torch.float64)

# ---- reference: srs-benchmark's own step(), iterated -------------------------------------
ref_r, ref_state = [], []
state = torch.zeros(B, 3, dtype=torch.float64)
for t in range(T):
    last_s = state[:, 0].clamp(cfg.s_min, fc.S_MAX)
    last_ss = state[:, 1].clamp(fc.S_MIN, fc.S_MAX)
    last_d = state[:, 2].clamp(fc.D_MIN, fc.D_MAX)
    is_first = (state == 0).all(dim=1)
    r = model.forgetting_curve(dt_t[:, t], last_s, last_ss, last_d)
    ref_r.append(torch.where(is_first, torch.full_like(r, float("nan")), r))
    X = torch.stack([dt_t[:, t], rt_t[:, t]], dim=1)
    state = model.step(X, state)
    ref_state.append(state.clone())

# ---- ours: the same sequences through rwkv/fsrs_core.py ----------------------------------
w = w_ref.expand(B, fc.N_PARAMS).contiguous()
our_r, our_state = [], []
s = ss = d = None
for t in range(T):
    if s is None:
        r = torch.full((B,), float("nan"), dtype=torch.float64)
        s, ss, d = fc.init_state(rt_t[:, t], w)
    else:
        r, s, ss, d = fc.step(s, ss, d, dt_t[:, t], rt_t[:, t], w)
    our_r.append(r)
    our_state.append(torch.stack([s, ss, d], dim=1))

# ---- compare -----------------------------------------------------------------------------
def report(name, a, b):
    a = torch.stack(a) if isinstance(a, list) else a
    b = torch.stack(b) if isinstance(b, list) else b
    m = ~(torch.isnan(a) | torch.isnan(b))
    if m.sum() == 0:
        print("  %-22s no comparable elements" % name)
        return 1
    diff = (a[m] - b[m]).abs()
    rel = diff / b[m].abs().clamp(min=1e-12)
    ok = rel.max().item() < 1e-5
    print("  %-22s n=%-7d max|abs| %.3e   max|rel| %.3e   %s"
          % (name, int(m.sum()), diff.max().item(), rel.max().item(),
             "OK" if ok else "*** MISMATCH"))
    return 0 if ok else 1


print("\ncomparing %d steps x %d sequences" % (T, B))
bad = 0
bad += report("retrievability", ref_r, our_r)
rs = torch.stack(ref_state)
os_ = torch.stack(our_state)
bad += report("S_long", rs[..., 0], os_[..., 0])
bad += report("S_short", rs[..., 1], os_[..., 1])
bad += report("difficulty", rs[..., 2], os_[..., 2])

# ---- bounded_w must be a real bound, checked at extremes, not at typical values -----------
lo = torch.tensor(fc.CLIP_LO, dtype=torch.float64)
hi = torch.tensor(fc.CLIP_HI, dtype=torch.float64)
z = torch.tensor([-1e4, -50.0, 0.0, 50.0, 1e4], dtype=torch.float64).unsqueeze(-1).expand(
    5, fc.N_PARAMS).contiguous()
wb = fc.bounded_w(z, lo, hi)
in_range = bool(((wb >= lo - 1e-9) & (wb <= hi + 1e-9)).all())
mono = bool((wb[..., 1] >= wb[..., 0]).all() and (wb[..., 2] >= wb[..., 1]).all()
            and (wb[..., 3] >= wb[..., 2]).all() and (wb[..., 26] >= wb[..., 25]).all())
print("  %-22s in-range %s   monotone %s"
      % ("bounded_w (extremes)", "OK" if in_range else "*** ESCAPES",
         "OK" if mono else "*** VIOLATED"))
bad += 0 if (in_range and mono) else 1
assert torch.isfinite(wb).all(), "bounded_w produced non-finite values"

print("\nRESULT: %s" % ("PORT VERIFIED" if not bad else "*** %d CHECK(S) FAILED" % bad))
sys.exit(1 if bad else 0)
