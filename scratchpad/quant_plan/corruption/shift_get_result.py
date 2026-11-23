"""Entry point that installs the raw token-shift dump probe and then runs a normal get_result eval.

Same shape as `state_get_result.py` / `tail_get_result.py`: the probe is imported and installed
BEFORE `rwkv.get_result` runs its model, and the whole run is guarded by `__main__` because
get_result spawns fetch workers with multiprocessing and on Windows the spawn re-imports this file
as `__mp_main__`.

    set RWKV_SHIFT_PROBE=1
    set RWKV_SHIFT_PROBE_DIR=scratchpad\\quant_plan\\corruption\\shift_probe_out
    set RWKV_SHIFT_PROBE_TAG=note_l2i8s4
    set RWKV_SHIFT_PROBE_STREAM=note
    .venv\\Scripts\\python.exe -u scratchpad\\quant_plan\\corruption\\shift_get_result.py ^
        --config scratchpad\\quant_plan\\corruption\\eval_shiftprobe.toml
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
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not (ROOT / "rwkv").is_dir():
        raise SystemExit(f"repo root not found from {HERE} (looked for {ROOT / 'rwkv'})")

    import shift_probe

    if not shift_probe.install():
        raise SystemExit("shift_get_result requires RWKV_SHIFT_PROBE=1 (refusing to run silently)")

    import rwkv.get_result as gr
    from rwkv.parse_toml import parse_toml

    config = parse_toml()
    print(f"[shift_get_result] model={config.MODEL_PATH} device={config.DEVICE} "
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
    shift_probe._dump()  # write the dump before the hard exit skips atexit
    if os.environ.get("RWKV_EXIT_HARD", "0") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
