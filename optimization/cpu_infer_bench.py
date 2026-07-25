"""CPU per-review inference cost vs model width -- THE metric the track-2 ablations exist
for (Andrew 2026-07-25: "I told you to do ablations hoping that fewer params -> faster CPU
inference in Anki"; state size is quantization's job, GPU training speed is incidental).

Times `SrsRWKVRnn.review()` -- the one-review-at-a-time recurrent path that ships -- on CPU
in fp32, carrying state across calls, exactly as Anki would. Also reports the analytic
per-review MAC count so the measurement can be read against the compute it should track.

Usage:
  python optimization/cpu_infer_bench.py                       # the track-2 width ladder
  python optimization/cpu_infer_bench.py --threads 3 --iters 400
  python optimization/cpu_infer_bench.py --arch path/to/architecture.py --label mine
"""
import argparse
import importlib.util
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LADDER = [
    ("A1 d=128 full", "scratchpad/track2_a1/architecture_d128_cmix1.py"),
    ("A9/A13 d=128", "scratchpad/track2_a9/architecture_d128_cmix1_user3_card2_note1.py"),
    ("A14 d=128 lora8", "scratchpad/track2_a14/architecture_d128_lora8.py"),
    ("A15 d=96", "scratchpad/track2_a15/architecture_d96_lora8.py"),
    ("A16 d=64", "scratchpad/track2_a16/architecture_d64_lora8.py"),
]


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("archmod", str(ROOT / path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEFAULT_ANKI_RWKV_CONFIG


def macs_per_review(cfg):
    """Analytic multiply-accumulates for ONE review through all 5 streams + heads.
    Per RWKV-7 layer: r/k/v/a/g projections + output proj (6*d^2), WKV state update and
    readout (~2*H*K*K = 2*d*K), channel mixer (2*d^2*cmf), LoRAs (small, counted)."""
    total = 0
    for _name, m in cfg.modules:
        d, H = m.d_model, m.n_heads
        K = d // H
        per_layer = 6 * d * d                      # r,k,v,a,g,out projections
        per_layer += 2 * H * K * K                 # WKV state update + readout
        per_layer += int(2 * d * d * m.channel_mixer_factor)
        per_layer += 2 * d * (m.decay_lora + m.a_lora + m.gate_lora + m.v0_mix_amt_lora)
        total += m.n_layers * per_layer
    d = cfg.d_model
    total += 92 * d * cfg.features_fc_mult + d * d * cfg.features_fc_mult   # input FC
    total += d * d * cfg.head_fc_mult                                        # head trunk
    return total


def bench(cfg, iters, warmup):
    import torch
    from rwkv.model.srs_model_rnn import SrsRWKVRnn

    torch.manual_seed(0)
    model = SrsRWKVRnn(cfg).float().eval()
    feats = torch.randn(1, 92)
    st = [None] * 5
    with torch.inference_mode():
        for i in range(warmup):
            out = model.review(feats, *st)
            st = list(out[-5:])
        times = []
        for i in range(iters):
            t0 = time.perf_counter()
            out = model.review(feats, *st)
            times.append(time.perf_counter() - t0)
            st = list(out[-5:])
    return statistics.median(times), sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--arch", default="")
    ap.add_argument("--label", default="custom")
    args = ap.parse_args()

    os.environ.setdefault("RWKV_NO_JIT", "1")
    import torch
    torch.set_num_threads(args.threads)

    ladder = [(args.label, args.arch)] if args.arch else LADDER
    print(f"CPU per-review inference, fp32, torch threads={args.threads}, "
          f"median of {args.iters} calls (warmup {args.warmup})\n")
    print(f"{'arch':<18}{'params':>10}{'MAC/review':>12}{'ms/review':>11}{'rev/s':>9}{'vs first':>10}")
    print("-" * 72)
    base = None
    for label, path in ladder:
        cfg = load_cfg(path)
        med, n = bench(cfg, args.iters, args.warmup)
        macs = macs_per_review(cfg)
        rate = 1.0 / med
        base = base or rate
        print(f"{label:<18}{n:>10,}{macs:>12,}{med*1000:>11.3f}{rate:>9.1f}{rate/base:>9.2f}x")


if __name__ == "__main__":
    main()
