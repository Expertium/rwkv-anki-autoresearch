"""Convert a training .pth checkpoint to the safetensors the Rust engine reads.
Mirrors export_rnn_trace.export_weights (float, contiguous, raw state_dict names).

Usage: .venv/Scripts/python.exe scratchpad/pth_to_sft.py <in.pth> <out.safetensors>
"""
import sys

import torch
from safetensors.torch import save_file

sd = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
flat = {k: v.detach().cpu().contiguous().float() for k, v in sd.items()}
save_file(flat, sys.argv[2])
print(f"{len(flat)} tensors -> {sys.argv[2]}")
