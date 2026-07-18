"""State-norm / NaN-site probe for A3's eval instability (2026-07-18).

Loads the A3 checkpoint (GRU curve head, d=128 A1 arch) exactly as get_result does,
fetches ONE NaN-skipped user's single eval batch through the real prepare pipeline,
and runs get_loss with forward hooks on every leaf module recording execution order,
output abs-max, and the FIRST module whose output goes non-finite.

Usage (env must carry the A3 construction flags -- see probe_a3_nan.cmd):
  python scratchpad/statenorm/probe_a3_nan.py <user_id> <bf16|fp32>

bf16 = reproduce the eval-time NaN and localize it.
fp32 = the A0-style precision question: still NaN -> weight-level; finite -> bf16 artifact.
"""

import multiprocessing
import os
import sys

import torch

sys.path.insert(0, os.getcwd())


def main():
    user_id = int(sys.argv[1])
    mode = sys.argv[2]
    assert mode in ("bf16", "fp32")
    # 3rd arg "jit": run the SCRIPTED forward, no hooks (hooks don't fire on ScriptModules).
    # Used by the state-clamp smoke: loss-only output, byte-comparable across env configs.
    jit_mode = len(sys.argv) > 3 and sys.argv[3] == "jit"
    noload = "noload" in sys.argv[3:]  # capture BEFORE the sys.argv rewrite below
    if not jit_mode:
        os.environ["RWKV_NO_JIT"] = "1"

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.data_fetcher import DataFetcher
    from rwkv.get_result import get_test_keys_batch
    from rwkv.model.srs_model import SrsRWKV
    from rwkv.parse_toml import parse_toml
    from rwkv.prepare_batch import prepare_data

    # parse_toml reads --config from argv (parse_known_args); PROBE_TOML overrides the
    # default A3 eval config (e.g. a d=32 toml for track-1 smokes)
    toml_path = os.environ.get("PROBE_TOML", "scratchpad/track2_a3/track2_a3_eval.toml")
    sys.argv = [sys.argv[0], "--config", toml_path]
    config = parse_toml()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32

    master_model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).to(config.DEVICE)
    model = (
        SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
        .selective_cast(dtype)
        .to(config.DEVICE)
    )
    if noload:
        # arch smoke: random weights, exercises the scripted forward + downcast chain
        print("NOLOAD: random init;", sum(p.numel() for p in master_model.parameters()), "params")
    else:
        print("Loading:", config.MODEL_PATH)
        master_model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))
    model.copy_downcast_(master_model, dtype=dtype)
    model.eval()
    del master_model
    torch.cuda.empty_cache()

    # ---- hooks: execution order + absmax + first-nonfinite site --------------------
    events = []          # (seq, qualified_name, absmax, nonfinite_kind)
    first_bad = []       # [(seq, name, kind)]

    def make_hook(name):
        def hook(_mod, _inp, out):
            tensors = []
            if torch.is_tensor(out):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [t for t in out if torch.is_tensor(t)]
            for t in tensors:
                if not t.is_floating_point():
                    continue
                f = t.float()
                amax = f.abs().max().item()
                kind = ""
                if torch.isinf(f).any().item():
                    kind = "INF"
                elif torch.isnan(f).any().item():
                    kind = "NAN"
                seq = len(events)
                events.append((seq, name, amax, kind))
                if kind and not first_bad:
                    first_bad.append((seq, name, kind))
        return hook

    if not jit_mode:
        n_hooked = 0
        for name, mod in model.named_modules():
            if name and len(list(mod.children())) == 0:  # leaf modules only
                mod.register_forward_hook(make_hook(name))
                n_hooked += 1
        print(f"hooked {n_hooked} leaf modules")

    # ---- fetch the user's batch through the real pipeline --------------------------
    all_keys = get_test_keys_batch(config, [user_id])
    batches = all_keys[user_id]
    print("user", user_id, "batches:", batches)

    with multiprocessing.Manager() as manager:
        task_queue = manager.Queue()
        batch_queue = manager.Queue()
        proc = multiprocessing.Process(
            target=prepare_data,
            args=(config.DATASET_LMDB_PATH, config.DATASET_LMDB_SIZE,
                  task_queue, batch_queue, 800000, 1234),
        )
        proc.start()
        fetcher = DataFetcher(task_queue=task_queue, out_queue=batch_queue)
        try:
            for batch_i, batch in enumerate(batches):
                fetcher.enqueue((f"probe-{user_id}-{batch_i}", [batch]))
            for batch_i in range(len(batches)):
                b = fetcher.get(f"probe-{user_id}-{batch_i}")
                b = b.to(config.DEVICE)
                if mode == "fp32":
                    b.start = b.start.float()
                    if b.labels.dtype == torch.bfloat16:
                        b.labels = b.labels.float()
                print(f"batch {batch_i}: T={b.start.shape}")
                events.clear()
                first_bad.clear()
                with torch.inference_mode():
                    stats = model.get_loss(b)
                print("get_loss ->", "None (NaN guard fired)" if stats is None else
                      f"ahead {stats.ahead_equalize_avg.item():.12f} imm {stats.imm_binary_equalize_avg.item():.12f}")

                if first_bad:
                    seq0, name0, kind0 = first_bad[0]
                    print(f"\nFIRST NON-FINITE: seq={seq0} module={name0} kind={kind0}")
                else:
                    print("\nno non-finite module outputs recorded")

                print("\n== last 15 modules before/at the first bad site ==")
                lo = max(0, (first_bad[0][0] - 12) if first_bad else len(events) - 15)
                hi = (first_bad[0][0] + 3) if first_bad else len(events)
                for seq, name, amax, kind in events[lo:hi]:
                    print(f"  [{seq:4d}] {name:70s} absmax={amax:12.4e} {kind}")

                print("\n== top-20 absmax across the whole forward ==")
                for seq, name, amax, kind in sorted(events, key=lambda e: -e[2])[:20]:
                    print(f"  [{seq:4d}] {name:70s} absmax={amax:12.4e} {kind}")
        finally:
            proc.terminate()
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
