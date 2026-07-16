"""Where does iter19's NaN first appear? (Andrew 2026-07-16)

Re-runs user 8902's chunks through the iter19 checkpoint in fp32 with a forward hook on
EVERY submodule, recording the first module whose output contains NaN/Inf while its inputs
were clean (= the creator), plus the propagation trail and per-module max|out| magnitudes.
Chunk 0 (finite in the probe) runs first as a magnitude baseline; chunk 1 is the NaN one.

Must run with RWKV_NO_JIT=1 (Python hooks don't fire on scripted modules) + the iter19 eval
env (RWKV_ZERO_FEATURES=22, H=2/K=16). fp32 weights; batch tensors upcast like the
RWKV_EVAL_CAST_FP32 shim. Fetch path = prepare_data's exact synchronous equivalent
(get_data -> prepare(target_len=800000, seed=1234) -- same args get_result.main passes).
"""

import os
import sys

import lmdb
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.config import RWKV_SUBMODULES  # noqa: E402
from rwkv.model.srs_model import SrsRWKV  # noqa: E402
from rwkv.prepare_batch import get_data, prepare  # noqa: E402

CKPT = os.path.join(REPO, "scratchpad/iter19_pbin025/iter19d_1638.pth")
DB = "F:/rwkv_lmdb/test_db_5k"
DB_SIZE = 250_000_000_000
CHUNKS = [
    ("chunk0_clean", (8902, 1, 1048576, 1979829)),
    ("chunk1_nan", (8902, 1048577, 2097152, 2006465)),
]

assert os.environ.get("RWKV_NO_JIT") == "1", "hooks need RWKV_NO_JIT=1"

print("stream order (rwkv_modules index -> name):", list(enumerate(RWKV_SUBMODULES)), flush=True)

model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to("cuda")
model.load_state_dict(torch.load(CKPT, weights_only=True))
model.eval()
print(f"loaded {CKPT} (fp32)", flush=True)

# ---- hooks ----------------------------------------------------------------
state = {"call": 0, "events": [], "max_abs": {}, "creator_reported": False}


def scan(t):
    if not torch.is_tensor(t) or not t.is_floating_point() or t.numel() == 0:
        return None
    nan = torch.isnan(t)
    inf = torch.isinf(t)
    return {
        "nan": bool(nan.any()),
        "inf": bool(inf.any()),
        "max": float(t.abs().amax()),
        "nan_mask": nan,
        "shape": tuple(t.shape),
    }


def first_bad_pos(s):
    # first flat index along dim 1 (T) if the tensor looks like (B,T,...) else flat
    m = s["nan_mask"]
    if not s["nan"]:
        return "inf-only"
    idx = m.reshape(m.shape[0], -1).any(dim=1) if m.dim() >= 2 else m
    if m.dim() >= 2:
        b = int(torch.nonzero(idx)[0])
        t_first = int(torch.nonzero(m[b].reshape(m.shape[1], -1).any(dim=1))[0]) if m.dim() >= 2 and m.shape[1] > 1 else 0
        frac = float(m.float().mean())
        return f"b={b} t_first={t_first} frac={frac:.4f}"
    return f"flat_first={int(torch.nonzero(m.reshape(-1))[0])}"


def mk_hook(name):
    def hook(mod, inp, out):
        state["call"] += 1
        outs = [scan(t) for t in (out if isinstance(out, tuple) else (out,))]
        outs = [o for o in outs if o]
        ins = [scan(t) for t in inp if torch.is_tensor(t)]
        ins = [i for i in ins if i]
        mx = max((o["max"] for o in outs), default=0.0)
        prev = state["max_abs"].get(name, 0.0)
        if mx > prev or mx != mx:  # track max (NaN-propagating)
            state["max_abs"][name] = mx
        out_bad = any(o["nan"] or o["inf"] for o in outs)
        if out_bad and len(state["events"]) < 200:
            in_bad = any(i["nan"] or i["inf"] for i in ins)
            max_in = max((i["max"] for i in ins), default=0.0)
            ev = {
                "call": state["call"], "name": name, "in_bad": in_bad,
                "max_in": max_in,
                "out_nan": any(o["nan"] for o in outs),
                "out_inf": any(o["inf"] for o in outs),
                "pos": first_bad_pos(next(o for o in outs if o["nan"] or o["inf"])),
            }
            state["events"].append(ev)
            if not in_bad and not state["creator_reported"]:
                state["creator_reported"] = True
                print(f"*** CREATOR: {name} (call {ev['call']}) -- inputs CLEAN "
                      f"(max|in|={max_in:.4g}), out nan={ev['out_nan']} inf={ev['out_inf']}, "
                      f"pos {ev['pos']}", flush=True)
    return hook


handles = [m.register_forward_hook(mk_hook(n)) for n, m in model.named_modules() if n]
print(f"hooked {len(handles)} modules", flush=True)

# ---- run ------------------------------------------------------------------
env = lmdb.open(DB, map_size=DB_SIZE)
baseline_max = {}
for label, key in CHUNKS:
    state["events"].clear()
    state["max_abs"] = {}
    state["creator_reported"] = False
    state["call"] = 0
    with env.begin(write=False) as txn:
        sample = get_data(txn, key, device="cpu")
    batch = prepare([sample], target_len=800000, seed=1234).to("cuda")
    batch.start = batch.start.float()
    if batch.labels.dtype == torch.bfloat16:
        batch.labels = batch.labels.float()
    print(f"\n===== {label} key={key} =====", flush=True)
    with torch.no_grad():
        stats = model.get_loss(batch)
    print(f"get_loss -> {'None (NaN guard fired)' if stats is None else 'finite stats'}", flush=True)

    if label == "chunk0_clean":
        baseline_max = dict(state["max_abs"])
        top = sorted(baseline_max.items(), key=lambda kv: -kv[1])[:12]
        print("clean-chunk top max|out| modules:", flush=True)
        for n, v in top:
            print(f"  {v:12.4g}  {n}", flush=True)
    else:
        if not state["events"]:
            print("NO NaN/Inf events this run (did not reproduce?)", flush=True)
        for ev in state["events"][:25]:
            base = baseline_max.get(ev["name"], float("nan"))
            print(f"  call {ev['call']:4d} {'CREATOR' if not ev['in_bad'] else 'trail  '} "
                  f"{ev['name']}  max|in|={ev['max_in']:.4g} nan={ev['out_nan']} "
                  f"inf={ev['out_inf']} pos[{ev['pos']}] clean-chunk max|out|={base:.4g}",
                  flush=True)
        if len(state["events"]) > 25:
            print(f"  ... {len(state['events']) - 25} more events", flush=True)
    del batch
    torch.cuda.empty_cache()

print("\nDIAG_DONE", flush=True)
