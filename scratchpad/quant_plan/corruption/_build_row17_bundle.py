"""Assemble the self-contained iter-17 (`l2i8s4`) bundle: everything needed to inspect, run and
verify the recipe, in one folder that can be zipped and sent.

WHAT IT PUTS WHERE, and why the layout is mirrored rather than flattened: the harness scripts
resolve their own paths from `Path(__file__).resolve().parents[N]` and import each other by bare
module name (`quant_sweep` -> `adaptive_train` lives in a SIBLING directory). Flattening them into
one folder would break every import, so `code/` reproduces the repo layout (`code/rwkv`,
`code/scratchpad/quant_plan`, `code/scratchpad/adaptive_wd`, `code/rust/rwkv-infer`,
`code/optimization`) and the scripts keep working when run from there.

Everything is a COPY: the repo is read only. Weights are copied by value; the deploy safetensors were
exported by the repo's own `export_weights_only.py` (so the recipe metadata inside them is the real
one, not a hand-written claim).

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_build_row17_bundle.py
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "scratchpad" / "row17_l2i8s4_bundle"
QP = ROOT / "scratchpad" / "quant_plan"

SKIP_DIRS = {"__pycache__", ".pytest_cache", "target", "target_offpath_old", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo"}

# (source relative to the repo, destination relative to the bundle)
FILES = [
    # --- weights -----------------------------------------------------------------------------
    ("pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth", "weights/rwkv_iter36_558.pth"),
    ("scratchpad/quant_plan/qat_it36l2i8s4/qat_124.pth", "weights/qat_l2i8s4_124.pth"),
    ("scratchpad/quant_plan/qat_it36ctl/ctl_124.pth", "weights/ctl_fp32_124.pth"),
    # --- traces (the Rust engine's input, exported by export_rnn_trace.export_user) -----------
    # `traces/`: the certified 3-user set, RE-EXPORTED for this lineage (the June-2026
    # reference/trace_user_* files carry a different feature recipe, so they are NOT shipped).
    ("scratchpad/quant_plan/trace_iter17_parity/trace_user_107.safetensors",
     "traces/trace_user_107.safetensors"),
    ("scratchpad/quant_plan/trace_iter17_parity/trace_user_107.json",
     "traces/trace_user_107.json"),
    ("scratchpad/quant_plan/trace_iter17_parity/trace_user_136.safetensors",
     "traces/trace_user_136.safetensors"),
    ("scratchpad/quant_plan/trace_iter17_parity/trace_user_136.json",
     "traces/trace_user_136.json"),
    # NOTE: user 156 is NOT here. The current label DB puts review 1142 of user 156 in the equalize
    # set, and the exporter's own per-review prediction dict has no entry for it, so `export_user`
    # aborts on its assertion. 107 and 136 export cleanly.
    # `traces_long/`: a long-history user (29,166 reviews, 23,455 equalized) from the corruption
    # roster, so the parity and finiteness checks also run on a history length where a recurrent
    # state has many steps in which to drift.
    ("scratchpad/quant_plan/trace_corrupt50/trace_user_5098.safetensors",
     "traces_long/trace_user_5098.safetensors"),
    ("scratchpad/quant_plan/trace_corrupt50/trace_user_5098.json",
     "traces_long/trace_user_5098.json"),
    # --- the training harness ---------------------------------------------------------------
    ("export_rnn_trace.py", "code/export_rnn_trace.py"),
    ("scratchpad/quant_plan/qat_decay.py", "code/scratchpad/quant_plan/qat_decay.py"),
    ("scratchpad/quant_plan/quant_sweep.py", "code/scratchpad/quant_plan/quant_sweep.py"),
    ("scratchpad/quant_plan/it36ladder_lib.py", "code/scratchpad/quant_plan/it36ladder_lib.py"),
    ("scratchpad/quant_plan/export_weights_only.py",
     "code/scratchpad/quant_plan/export_weights_only.py"),
    ("scratchpad/adaptive_wd/adaptive_train.py",
     "code/scratchpad/adaptive_wd/adaptive_train.py"),
    # --- the deploy-side measurement tools --------------------------------------------------
    ("scratchpad/quant_plan/rust_grid.py", "code/scratchpad/quant_plan/rust_grid.py"),
    ("scratchpad/quant_plan/rust_speed.py", "code/scratchpad/quant_plan/rust_speed.py"),
    ("scratchpad/quant_plan/rust_score.py", "code/scratchpad/quant_plan/rust_score.py"),
    ("scratchpad/quant_plan/rust_finite.py", "code/scratchpad/quant_plan/rust_finite.py"),
    ("scratchpad/quant_plan/py_rust_parity.py", "code/scratchpad/quant_plan/py_rust_parity.py"),
    ("scratchpad/quant_plan/rust_speed_idle_l2i8.json",
     "evidence/rust_speed_idle.json"),
    ("scratchpad/quant_plan/rust_finite_probe.json", "evidence/rust_finite_probe.json"),
    ("scratchpad/quant_plan/rust_score.json", "evidence/rust_score.json"),
    # --- the corruption (finiteness) gate ---------------------------------------------------
    ("scratchpad/quant_plan/corruption/check_corruption.py",
     "code/scratchpad/quant_plan/corruption/check_corruption.py"),
    ("scratchpad/quant_plan/corruption/finite_probe.py",
     "code/scratchpad/quant_plan/corruption/finite_probe.py"),
    ("scratchpad/quant_plan/corruption/probe_get_result.py",
     "code/scratchpad/quant_plan/corruption/probe_get_result.py"),
    ("scratchpad/quant_plan/corruption/corrupt50.json",
     "code/scratchpad/quant_plan/corruption/corrupt50.json"),
    ("scratchpad/quant_plan/corruption/export_corrupt50_traces.py",
     "code/scratchpad/quant_plan/corruption/export_corrupt50_traces.py"),
    ("scratchpad/quant_plan/corruption/export_parity_trace.py",
     "code/scratchpad/quant_plan/corruption/export_parity_trace.py"),
    ("scratchpad/quant_plan/corruption/users_parity2.json",
     "code/scratchpad/quant_plan/corruption/users_parity2.json"),
    ("scratchpad/quant_plan/corruption/corruption_corrupt_l2i8s4.json",
     "evidence/corruption_iter17_l2i8s4.json"),
    # --- the parity checks (see verification.md section 4) -----------------------------------
    ("scratchpad/quant_plan/py_rust_parity_iter17_plain.json",
     "evidence/parity/107_plain_rust_vs_python_rnn.json"),
    ("scratchpad/quant_plan/py_rust_parity_corrupt50_plain.json",
     "evidence/parity/long_user_plain_rust_vs_python_rnn.json"),
    ("scratchpad/quant_plan/py_rust_parity_corrupt50_l2i8s4.json",
     "evidence/parity/long_user_l2i8s4_rust_vs_unquantized_python.json"),
    ("scratchpad/quant_plan/rust_finite_iter17_l2i8s4.json",
     "evidence/rust_finite_iter17_l2i8s4.json"),
    ("scratchpad/quant_plan/rust_grid_iter17_l2i8s4.json",
     "evidence/rust_grid_iter17_l2i8s4.json"),
    ("scratchpad/quant_plan/rust_speed_bundle.json",
     "evidence/rust_speed_bundle.json"),
    ("scratchpad/quant_plan/trace_iter17_parity/parity_export.json",
     "evidence/parity/parity_export.json"),
    # --- the run that produced iter 17 ------------------------------------------------------
    ("scratchpad/quant_plan/qat_it36l2i8s4/train.toml", "evidence/run/train.toml"),
    ("scratchpad/quant_plan/qat_it36l2i8s4/qat_env.json", "evidence/run/qat_env.json"),
    ("scratchpad/quant_plan/qat_it36l2i8s4/train.log", "evidence/run/train.log"),
    ("scratchpad/quant_plan/qat_it36l2i8s4smoke/train.log", "evidence/run/smoke_train.log"),
    ("scratchpad/quant_plan/qat_it36l2i8s4smoke/train.toml", "evidence/run/smoke_train.toml"),
    # --- the evaluation that produced the gate numbers ---------------------------------------
    ("scratchpad/quant_plan/quant_sweep_it36l2i8s4.json", "evidence/sweep_iter17_l2i8s4.json"),
    ("scratchpad/quant_plan/eval_ql2i8s4_it36l2i8s4.toml", "evidence/eval/eval_config.toml"),
    ("scratchpad/quant_plan/users_ql2i8s4_it36l2i8s4.json",
     "evidence/eval/eval_users_750.json"),
    ("scratchpad/quant_plan/roster750_seed20261119.json",
     "evidence/eval/roster750_seed20261119.json"),
    ("result/adw_ql2i8s4_it36l2i8s4.jsonl", "evidence/eval/750_users_ahead_logloss.jsonl"),
    ("result/adwP_ql2i8s4_it36l2i8s4.jsonl", "evidence/eval/750_users_imm_logloss.jsonl"),
    ("result/adw_corrupt_l2i8s4.jsonl", "evidence/corruption/50_users_ahead_logloss.jsonl"),
    ("result/adwP_corrupt_l2i8s4.jsonl", "evidence/corruption/50_users_imm_logloss.jsonl"),
    # --- the contract the recipe is measured against ----------------------------------------
    ("optimization/DEPLOY_FUNCTIONS.md", "code/optimization/DEPLOY_FUNCTIONS.md"),
    ("optimization/PROTOCOL.md", "code/optimization/PROTOCOL.md"),
    ("rwkv/get_result_config_base5k.toml", "code/rwkv/get_result_config_base5k.toml"),
]

DIRS = [
    ("rwkv", "code/rwkv"),
    ("rust/rwkv-infer", "code/rust/rwkv-infer"),
]

# Evidence assembled by this script from the repo's own artifacts (not hand-typed).
GENERATED = {}


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def copy_dir(src: Path, dst: Path):
    n = 0
    for p in src.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in SKIP_SUFFIX or p.is_dir():
            continue
        rel = p.relative_to(src)
        copy_file(p, dst / rel)
        n += 1
    return n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if B.exists() and "--clean" in sys.argv:
        shutil.rmtree(B)
    B.mkdir(parents=True, exist_ok=True)

    missing = []
    for src_rel, dst_rel in FILES:
        src = ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        copy_file(src, B / dst_rel)
    for src_rel, dst_rel in DIRS:
        src = ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        n = copy_dir(src, B / dst_rel)
        print(f"copied {n:4d} files  {src_rel} -> {dst_rel}")
    if missing:
        print(f"\nMISSING (not copied): {missing}")

    # ---- the manifest --------------------------------------------------------------------
    rows = []
    for p in sorted(B.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.md":
            rows.append((p.relative_to(B).as_posix(), p.stat().st_size, sha256(p)))
    total = sum(r[1] for r in rows)
    lines = [
        "# MANIFEST",
        "",
        "Every file in this bundle, with its size and SHA-256, so the recipient can check that what",
        "they received is what was sent. Generated by the repo's own bundle builder; nothing here is",
        "hand-typed.",
        "",
        f"`{len(rows)}` files, `{total / 1e6:.2f}` MB total (uncompressed).",
        "",
        "| file | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for rel, size, dig in rows:
        lines.append(f"| `{rel}` | {size:,} | `{dig}` |")
    lines += [
        "",
        "## NOT included, and why",
        "",
        "* **The datasets.** `F:/rwkv_lmdb/train_db_5k_h1_adaptive92`,",
        "  `F:/rwkv_lmdb/test_db_5k_adaptive92` and `F:/rwkv_lmdb/label_filter_db_id_e2s` are multi-GB",
        "  LMDBs. Re-running the 124-step QAT decay or the 750-user eval needs them; running and",
        "  scoring the Rust engine does NOT (the traces are included).",
        "* **Optimizer states** (`qat_optim_124.pth`, `rwkv_iter36_optim_558.pth`, ~1.9 MB each). They",
        "  are only needed to RESUME a run, not to reproduce or inspect one.",
        "* **The Python interpreter environment.** See `verification.md` for the package versions the",
        "  numbers were produced with.",
        "* **`RWKV_CUDA.cp312-win_amd64.pyd`** IS included (1.46 MB, the compiled `torch.ops.rwkv`",
        "  extension used for the fused QAT kernels). It is Windows/py3.12 only; its source is in",
        "  `code/rwkv/model/csrc/` and it can be rebuilt with `nvcc`.",
        "",
    ]
    (B / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(rows)} files, {total / 1e6:.2f} MB -> {B}")
    print(f"manifest: {B / 'MANIFEST.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
