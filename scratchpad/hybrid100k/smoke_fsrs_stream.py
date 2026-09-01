"""Properties of the FSRS card core (rwkv/fsrs_stream.py), before it is wired into anything.

The math is already verified against srs-benchmark (smoke_fsrs_port.py, machine precision).
What is unverified is the WRAPPER: initialization, the first-review sentinel, state carry, and
skip handling. Each of these fails silently rather than loudly, so each gets a check.

  1. UNTRAINED == STOCK FSRS-7. The emitter's weight is zero and its bias decodes to INIT_W, so
     an untrained core emits FSRS-7's own default parameters for every input. Combined with the
     verified port this means the untrained core IS stock FSRS-7 -- a strong initialization and
     a strong sanity check.
  2. THE WRITEBACK IS NOT ZERO. This is the decision most likely to be "corrected" later by
     someone applying the house zero-init style. With W_out = 0 the gradient to the emitter is
     identically zero and the core can never start learning; the run would return a clean null
     that looks like evidence about FSRS and is actually evidence about initialization.
  3. STATE CARRY. Running a sequence in one call must equal splitting it and carrying the state.
     Training chunks sequences and deploy is one review at a time, so if this does not hold the
     two paths compute different functions -- the §9 three-way-parity failure mode.
  4. THE FIRST-REVIEW SENTINEL IS PER ELEMENT. A batch mixing fresh and non-fresh cards must
     take the init path only for the fresh ones. A whole-tensor Python branch would pass a
     uniform batch and fail here, which is exactly why the test uses a mixed one.
  5. SKIP DOES NOT ADVANCE STATE, but still produces an output (probe rows need both).
  6. STATE SIZE is 3 + n_free floats, against the champion's 2,880.

CPU-only, seconds.
Usage: .venv/Scripts/python.exe scratchpad/hybrid100k/smoke_fsrs_stream.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

torch.set_num_threads(1)
from rwkv import fsrs_core as fc        # noqa: E402
from rwkv import fsrs_stream as fs      # noqa: E402

D = 32
fails = []


def check(name, ok, detail=""):
    print("  %-46s %s%s" % (name, "OK" if ok else "*** FAIL", "" if ok else "  " + detail))
    if not ok:
        fails.append(name)


def make(n_free=0, seed=0):
    torch.manual_seed(seed)
    return fs.FsrsCardCore(D, n_free=n_free).double()


# ---- 1. untrained == stock FSRS-7 ----------------------------------------------------------
core = make()
x = torch.randn(7, D, dtype=torch.float64) * 3.0          # arbitrary, even large, inputs
w = core.params_from(x)
w0 = torch.tensor(fc.INIT_W, dtype=torch.float64)
err = (w - w0).abs().max().item()
check("untrained emitter reproduces FSRS-7 INIT_W", err < 1e-3, "max|diff| %.2e" % err)
check("...for every input (zero weight)", bool((w - w[0]).abs().max() < 1e-12))

# ---- 2. the writeback is deliberately not zero ---------------------------------------------
wb = core.writeback.weight.abs().max().item()
check("writeback init is NON-zero (see docstring)", wb > 1e-3, "max|W_out| %.2e" % wb)
check("emitter weight init IS zero", core.emit.weight.abs().max().item() == 0.0)

# ---- gradient actually reaches the emitter -------------------------------------------------
core.zero_grad()
st = fs.zero_state((5,), 0, torch.float64, x.device)
xin = torch.randn(5, D, dtype=torch.float64, requires_grad=True)
out, _r, _st = core.review(xin, torch.full((5,), 3.0, dtype=torch.float64),
                           torch.full((5,), 3.0, dtype=torch.float64), st)
out.sum().backward()
g = core.emit.weight.grad
check("gradient reaches the emitter", g is not None and g.abs().max().item() > 0,
      "max|grad| %.2e" % (0.0 if g is None else g.abs().max().item()))

# ---- 3. state carry ------------------------------------------------------------------------
for n_free in (0, 5):
    core = make(n_free=n_free, seed=1)
    B, T = 6, 9
    xs = torch.randn(B, T, D, dtype=torch.float64)
    t = torch.rand(B, T, dtype=torch.float64) * 30.0
    rat = torch.randint(1, 5, (B, T)).double()
    whole, s_whole = fs.run_sequence(core, xs, t, rat, None, None)
    a, s_a = fs.run_sequence(core, xs[:, :4], t[:, :4], rat[:, :4], None, None)
    b, s_b = fs.run_sequence(core, xs[:, 4:], t[:, 4:], rat[:, 4:], None, s_a)
    split = torch.cat([a, b], dim=1)
    e1 = (whole - split).abs().max().item()
    e2 = (s_whole - s_b).abs().max().item()
    check("state carry, n_free=%d (outputs)" % n_free, e1 < 1e-10, "max|diff| %.2e" % e1)
    check("state carry, n_free=%d (final state)" % n_free, e2 < 1e-10, "max|diff| %.2e" % e2)

# ---- 4. the first-review sentinel is per element -------------------------------------------
core = make(seed=2)
x1 = torch.randn(4, D, dtype=torch.float64)
rat = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
t1 = torch.full((4,), 10.0, dtype=torch.float64)
fresh = fs.zero_state((4,), 0, torch.float64, x1.device)
_o, _r, st_all_fresh = core.review(x1, t1, rat, fresh)
# a MIXED batch: elements 0 and 2 fresh, 1 and 3 carrying the state just produced
mixed = st_all_fresh.clone()
mixed[0] = 0.0
mixed[2] = 0.0
_o, _r, st_mixed = core.review(x1, t1, rat, mixed)
same_as_fresh = (st_mixed - st_all_fresh).abs().max(dim=-1)[0] < 1e-12
check("fresh elements take the init path",
      bool(same_as_fresh[0]) and bool(same_as_fresh[2]))
check("stateful elements take the update path",
      (not bool(same_as_fresh[1])) and (not bool(same_as_fresh[3])))

# ---- 5. skip does not advance state, but does produce output -------------------------------
core = make(seed=3)
B, T = 4, 5
xs = torch.randn(B, T, D, dtype=torch.float64)
t = torch.rand(B, T, dtype=torch.float64) * 20.0
rat = torch.randint(1, 5, (B, T)).double()
skip = torch.zeros(B, T)
skip[:, 2] = 1.0                                   # every card's 3rd row is a probe
o_skip, s_skip = fs.run_sequence(core, xs, t, rat, skip, None)
keep_idx = [0, 1, 3, 4]
o_ref, s_ref = fs.run_sequence(core, xs[:, keep_idx], t[:, keep_idx], rat[:, keep_idx],
                               None, None)
e = (s_skip - s_ref).abs().max().item()
check("skipped rows do not advance state", e < 1e-10, "max|diff| %.2e" % e)
check("skipped rows still produce an output",
      bool(torch.isfinite(o_skip[:, 2]).all()) and bool(o_skip[:, 2].abs().sum() > 0))

# ---- 6. state size -------------------------------------------------------------------------
for n_free in (0, 5):
    st = fs.zero_state((1,), n_free, torch.float64, torch.device("cpu"))
    check("card state is %d floats (champion: 2880)" % (3 + n_free),
          st.shape[-1] == 3 + n_free)

# ---- flag parsing --------------------------------------------------------------------------
for val, want_on, want_n in (("", False, -1), ("0", True, 0), ("5", True, 5)):
    os.environ["RWKV_FSRS_CARD"] = val
    if val == "":
        del os.environ["RWKV_FSRS_CARD"]
    check("RWKV_FSRS_CARD=%-3r -> on=%-5s n_free=%d" % (val, want_on, want_n),
          fs.is_on() == want_on and fs.n_free_dims() == want_n)
os.environ.pop("RWKV_FSRS_CARD", None)

print("\nRESULT: %s" % ("ALL CHECKS PASSED" if not fails
                        else "*** %d FAILED: %s" % (len(fails), ", ".join(fails))))
sys.exit(1 if fails else 0)
