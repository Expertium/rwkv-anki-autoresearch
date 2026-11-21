"""Entry point that installs the finiteness probe and then runs a normal get_result eval.

Usage (the driver does this for you -- see check_corruption.py):

    set RWKV_FINITE_PROBE=1
    set RWKV_FINITE_PROBE_DIR=scratchpad\\quant_plan\\corruption\\probe_out\\<tag>
    .venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\corruption\\probe_get_result.py ^
        --config scratchpad\\quant_plan\\eval_<tag>.toml

This mirrors `rwkv/get_result.py`'s own `__main__` block, but calls the ALREADY-IMPORTED module so
that the probe's patches of `get_result.get_benchmark_info` are the ones `run()` resolves (a
`runpy.run_module` here would re-execute the module into a fresh namespace and silently restore the
unpatched per-user hook).

Everything runs under `if __name__ == "__main__":` because get_result spawns fetch workers with
multiprocessing: on Windows the spawn re-imports this file as `__mp_main__`, and an unguarded
module-level eval would make every worker re-run the whole evaluation (observed: two workers wrote
probe `_summary` lines with no data and the parent never finished).
"""

import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # corruption -> quant_plan -> scratchpad -> repo root


def main() -> int:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if str(ROOT) not in sys.path:  # launched by path, so cwd is not on sys.path: `import rwkv` needs it
        sys.path.insert(0, str(ROOT))
    if not (ROOT / "rwkv").is_dir():
        raise SystemExit(f"repo root not found from {HERE} (looked for {ROOT / 'rwkv'})")

    import finite_probe

    if not finite_probe.install():
        raise SystemExit("probe_get_result requires RWKV_FINITE_PROBE=1 (refusing to run silently)")

    import rwkv.get_result as gr
    from rwkv.parse_toml import parse_toml

    config = parse_toml()
    print(f"[probe_get_result] model={config.MODEL_PATH} device={config.DEVICE} "
          f"dtype={config.DTYPE}", flush=True)
    try:
        gr.main(config)
    except Exception:
        traceback.print_exc()
        if os.environ.get("RWKV_EXIT_HARD", "0") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    if os.environ.get("RWKV_EXIT_HARD", "0") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
