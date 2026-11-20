"""Export a .pth checkpoint to the deploy safetensors artifact (weights + recipe metadata) only.

Reuses `export_rnn_trace.py`'s own `export_weights()` -- the function the deploy artifact has always
come from -- so the stream order, interleave placement and state-clamp recipe facts are read off the
model object and written into the safetensors metadata exactly as a normal export does. The trace
export is skipped: traces for the parity users already exist in `reference/` and the speed
measurement does not need new ones.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/export_weights_only.py <in.pth> <out_dir> <out.safetensors>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_ID_FEATURES", "0")
os.environ.setdefault("RWKV_REAL_CYCLES", "0")
os.environ.setdefault("RWKV_FINAL_LAYOUT", "0")
os.environ.setdefault("RWKV_E2S_PUBLISHED", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

inp, out_dir, out_name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

import export_rnn_trace as e  # noqa: E402

e.MODEL_PATH = str((ROOT / inp) if not Path(inp).is_absolute() else Path(inp))
e.OUT_DIR = out_dir if out_dir.is_absolute() else ROOT / out_dir
e.WEIGHTS_SFT = out_name
e.OUT_DIR.mkdir(parents=True, exist_ok=True)
e.export_weights()
print(f"weights: {e.OUT_DIR / out_name}")
