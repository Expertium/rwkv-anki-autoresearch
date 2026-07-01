"""Export ONLY the champion weights -> reference/<RWKV_CHAMP_SFT> (no per-user trace export).
The Rust deploy eval reuses existing trace_user_{u}.safetensors (their feature/routing INPUTS are
weight-independent), so for a new champion we only need to re-export its weights. architecture.py must
match the checkpoint (the d=32 champion arch is unchanged by HP tuning). Run from the repo root.
Env: RWKV_CHAMP_CKPT (the .pth), RWKV_CHAMP_SFT (output safetensors name under reference/)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_CHAMP_CKPT", "scratchpad/tuner/decay15/decay15_640.pth")
os.environ.setdefault("RWKV_CHAMP_SFT", "champ_decay15.safetensors")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for export_rnn_trace
import export_rnn_trace as ert  # reads RWKV_CHAMP_CKPT/SFT at import for MODEL_PATH/WEIGHTS_SFT

ert.OUT_DIR.mkdir(exist_ok=True)
ert.export_weights()
print(f"exported weights from {ert.MODEL_PATH} -> reference/{ert.WEIGHTS_SFT}")
