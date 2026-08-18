"""iter 55 smoke: RWKV_RGATE (FSRS-form retrievability gating of the delta-rule rate `a`).

WHY THIS EXISTS ALONGSIDE THE parity_train_vs_rnn.py CASE. That harness is SINGLE-STREAM: it
proves the gate's arithmetic agrees between the training and deploy recurrences. It cannot see
the part of this lever that is genuinely new and genuinely risky -- the PLUMBING. `log_dt` is the
first RAW INPUT FEATURE ever threaded into the recurrence, and under RWKV_INTERLEAVE it has to be
gathered with each split's own canonical indices so it lines up row-for-row with `x`. An
off-by-one there is invisible: the model still trains, still evaluates, and simply gates every
review on some OTHER review's elapsed time.

Checks, in the order the failures actually happen:

 1. **INERT WHEN OFF** -- no `rgate*` key in the state_dict. An unconditional Parameter would give
    the 421-key champion checkpoint extra keys and break `load_state_dict(strict=True)`.
 2. **ON AT INIT == OFF, BIT-FOR-BIT.** `rgate_gain` is zero-init, so the claim "even switched on,
    this starts exactly at the champion" is checkable rather than merely asserted -- and it is the
    single strongest statement that the plumbing perturbs nothing. Requires the two models to
    share weights for every COMMON parameter, which a plain `manual_seed` cannot give (the ON
    model has more tensors, so a single RNG stream desynchronises). Params are therefore filled
    per-name from a name-seeded generator.
 3. **NOT VACUOUS** -- with a real non-zero gain the output must MOVE. Without this, check 2 passes
    on a lever that is wired to nothing, which is exactly the silent-null failure the interleave
    smoke was written to prevent.
 4. **THE RECOVERED log_dt IS PHYSICALLY REAL.** The gate inverts data_processing's
    standardization to get natural-log elapsed SECONDS. On a real chunk that number must look like
    log-seconds (median inside [log 60 s, log 90 d]) and must contain exact 0.0 rows -- the -1
    "no previous review" sentinel, which `scale_elapsed_seconds` maps to 0 before standardizing.
    Wire the gate to the wrong column (scaled_elapsed_DAYS is column 0, one slot away) and this
    check fails loudly instead of training a subtly wrong model for 6 hours.
 5. **GRADIENTS REACH ALL THREE TENSORS** -- the index_select detour must not detach the graph,
    and a gate that cannot learn would measure as a null for the wrong reason.

Real chunk, real prepare(), probes included -- same source as smoke_interleave.py. Children run
JIT-ON so construction IS the TorchScript compile. Each env gets its own subprocess (old-style
ScriptModule bakes the first construction's flags into the compiled class).

Run:  .venv\\Scripts\\python.exe scratchpad/parity3/smoke_rgate.py
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import json, math, os, zlib, torch
import lmdb
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.data_processing import CARD_FEATURE_COLUMNS, STATISTICS
from rwkv.model.srs_model import SrsRWKV
from rwkv.prepare_batch import get_data, prepare

OUT = os.environ["SMOKE_OUT"]
GAIN = float(os.environ.get("SMOKE_GAIN", "0"))      # 0 = inertness arm, !=0 = non-vacuity arm
WANT_GRAD = os.environ.get("SMOKE_GRAD", "0") == "1"

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
# Fill each parameter from a generator seeded by its NAME. A single manual_seed would desync the
# moment the ON model adds tensors, and then "bit-identical" could never hold for an honest
# reason -- the two models would simply have different weights everywhere.
with torch.no_grad():
    for name, p in model.named_parameters():
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()) & 0x7FFFFFFF)
        p.copy_(torch.empty_like(p).normal_(0.0, 0.1, generator=g))
    # the lever under test: gain is zero-init in the real model, so the inertness arm restores
    # that after the blanket randomization above, and the non-vacuity arm sets it deliberately.
    n_gain = 0
    for name, p in model.named_parameters():
        if name.endswith("rgate_gain"):
            p.fill_(GAIN)
            n_gain += 1
        elif name.endswith("rgate_log_s.bias"):
            # keep log_s in the responsive band (see rwkv_model.py); randomized to ~0 it would
            # drive rhat to ~1e-4 for every real gap and the gate would degenerate to a constant
            p.fill_(float(STATISTICS["elapsed_seconds_mean"]))
model = model.float()
model.eval()

rgate_keys = sorted(k for k in model.state_dict() if "rgate" in k)
n_params = sum(p.numel() for p in model.parameters())
print(f"RGATE_KEYS {len(rgate_keys)} GAIN_TENSORS {n_gain} PARAMS {n_params}")

env = lmdb.open("train_db_5k_h1", map_size=400_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    batches = json.loads(txn.get(b"101_batches"))
    b = min(batches, key=lambda x: x[2])
    data = get_data(txn, (101, b[0], b[1], b[2]), device="cpu")
env.close()

pb = prepare([data], seed=1234, probe_density=0.08)
pb = pb.to("cpu")

# ---- check 4: is the recovered log_dt actually log-seconds? ----------------------------------
col = CARD_FEATURE_COLUMNS.index("scaled_elapsed_seconds")
log_dt = pb.start.float()[:, col] * STATISTICS["elapsed_seconds_std"] + STATISTICS["elapsed_seconds_mean"]
q = torch.quantile(log_dt, torch.tensor([0.0, 0.5, 1.0]))
# ⚠ NOT an exact-zero test, and the reason is worth knowing: the LMDB stores features in
# BFLOAT16. `scale_elapsed_seconds` maps the -1 "no previous review" sentinel to 0, i.e. a
# standardized -1.9117082534, which bf16 rounds to -1.9140625; un-standardizing multiplies that
# error by std=5.21, so a first review recovers as log_dt = -0.01227, not 0.0. Substantively
# identical (dt = 0.99 s, so rhat ~ 1 and the gate contributes ~0 exactly as intended), but any
# `== 0` test would fail forever. The same +/-0.02 log-space quantization applies to every row --
# ~2% in dt, immaterial to a smooth retrievability function.
SENTINEL = (0.0 - STATISTICS["elapsed_seconds_mean"]) / STATISTICS["elapsed_seconds_std"]
SENTINEL = SENTINEL * STATISTICS["elapsed_seconds_std"] + STATISTICS["elapsed_seconds_mean"]
n_sentinel = int(((log_dt - SENTINEL).abs() < 0.05).sum())
print(f"LOG_DT col={col} min={q[0]:.3f} median={q[1]:.3f} max={q[2]:.3f} "
      f"first_review_rows={n_sentinel}/{log_dt.numel()}")
assert math.log(60.0) < float(q[1]) < math.log(90 * 86400.0), (
    f"median log_dt {float(q[1]):.3f} is not plausible for log-SECONDS -- wrong column or wrong "
    f"standardization constants (scaled_elapsed_days is column 0, one slot away)")
assert n_sentinel > 0, (
    "no rows near the first-review sentinel -- expected log_dt ~ 0 for a card's first review")

torch.set_grad_enabled(WANT_GRAD)
out = model.forward_batch(
    pb.start.float(), pb.sub_gather, pb.sub_gather_lens,
    pb.time_shift_selects, pb.skips, pb.num_data,
)
flat = torch.cat([o.float().flatten() for o in out])
if WANT_GRAD:
    flat.square().mean().backward()
    missing = [n for n, p in model.named_parameters()
               if "rgate" in n and (p.grad is None or not torch.isfinite(p.grad).all()
                                    or p.grad.abs().max().item() == 0.0)]
    assert not missing, f"rgate params with no/zero/non-finite grad: {missing}"
    print(f"GRAD_OK all {len([n for n,_ in model.named_parameters() if 'rgate' in n])} "
          f"rgate tensors have finite non-zero grads")
torch.save({"flat": flat.detach()}, OUT)
print(f"rows={pb.start.size(0)} checksum={flat.double().sum().item():.10f} "
      f"scale={flat.abs().max().item():.4f}")
assert flat.abs().max().item() > 1e-3, "outputs ~zero -- comparison would be vacuous"
print("CHILD_OK")
"""

ARCH = {
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_GRU_HEAD": "3",
    "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_ZERO_FEATURES": "22",
    "RWKV_NO_AHEAD_RESIDUAL": "1",
    "RWKV_INTERLEAVE": "1",
}


# Every env var this smoke controls. Each arm states its OWN value for all of them, so an arm that
# omits one gets it UNSET rather than whatever the caller happened to export.
_SMOKE_VARS = ("RWKV_RGATE", "SMOKE_GAIN", "SMOKE_GRAD")


def run_child(tag, extra_env, out_path):
    env = dict(os.environ, PYTHONPATH=REPO, SMOKE_OUT=out_path, **ARCH)
    # ★ HERMETIC: strip the smoke's own vars before applying this arm's.
    # `run_iter55.cmd` does `set RWKV_RGATE=card` BEFORE calling this script, and the original
    # `dict(os.environ, **extra_env)` let the OFF arm inherit it. Both arms then built the SAME
    # gated model: the param check failed ("rgate keys present with the flag OFF"), and worse, the
    # inertness check compared two gated models and passed VACUOUSLY at 0.000e+00. A test that
    # reads its control's configuration from the ambient environment is not a control.
    for v in _SMOKE_VARS:
        env.pop(v, None)
    env.update(extra_env)
    env.pop("RWKV_NO_JIT", None)   # JIT ON: construction must compile
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    print(f"--- {tag}")
    for ln in lines:
        if any(k in ln for k in ("RGATE_KEYS", "LOG_DT", "GRAD_OK", "rows=", "CHILD_OK")):
            print("    " + ln)
    if p.returncode != 0 or not any("CHILD_OK" in ln for ln in lines):
        print("FAILED CHILD, full output:")
        print(p.stdout + p.stderr)
        sys.exit(1)
    return [ln for ln in lines if "RGATE_KEYS" in ln or "PARAMS" in ln]


def main():
    import torch

    if not os.path.isdir(os.path.join(REPO, "train_db_5k_h1")):
        print("SKIP: train_db_5k_h1 not present")
        return 0

    tmp = tempfile.mkdtemp(prefix="rgate_smoke_")
    f_off = os.path.join(tmp, "off.pt")
    f_on0 = os.path.join(tmp, "on_gain0.pt")
    f_on1 = os.path.join(tmp, "on_gain.pt")

    info_off = run_child("1. flag OFF (champion path)", {}, f_off)
    info_on = run_child("2. flag ON, gain=0 (must be inert)",
                        {"RWKV_RGATE": "card"}, f_on0)
    run_child("3. flag ON, gain=0.8 (must move) + grads",
              {"RWKV_RGATE": "card", "SMOKE_GAIN": "0.8", "SMOKE_GRAD": "1"}, f_on1)

    off = torch.load(f_off)["flat"]
    on0 = torch.load(f_on0)["flat"]
    on1 = torch.load(f_on1)["flat"]

    n_off = int(info_off[0].split()[1])
    n_on = int(info_on[0].split()[1])
    p_off = int(info_off[0].split()[5])
    p_on = int(info_on[0].split()[5])
    print(f"\nstate_dict rgate keys: OFF {n_off}, ON {n_on}")
    print(f"params: OFF {p_off}, ON {p_on}, delta +{p_on - p_off}")

    ok = True
    if n_off != 0:
        print("FAIL: rgate keys present with the flag OFF"); ok = False
    if n_on != 8:
        print(f"FAIL: expected 8 rgate keys ON (2 gated layers x [log_s.w, log_s.b, d.w, gain])"
              f", got {n_on}"); ok = False
    # 2 gated layers x (80 log_s weight + 1 log_s bias + 80 d weight + 1 gain)
    if p_on - p_off != 324:
        print(f"FAIL: expected +324 params, got +{p_on - p_off}"); ok = False

    d_inert = (off - on0).abs().max().item()
    d_move = (on0 - on1).abs().max().item()
    print(f"\nON@gain=0 vs OFF   max|delta| = {d_inert:.3e}   (must be exactly 0)")
    print(f"gain=0.8 vs gain=0 max|delta| = {d_move:.3e}   (must be > 0)")
    if d_inert != 0.0:
        print("FAIL: the lever is NOT inert at init -- plumbing perturbs the champion path")
        ok = False
    if d_move <= 1e-4:
        print("FAIL: a real gain did not move the output -- the lever is wired to nothing")
        ok = False

    print("\nRGATE_" + ("ALL_PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
