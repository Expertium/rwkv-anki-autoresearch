from pathlib import Path
import torch
for p in Path("pretrain/rwkv").rglob("*.pth"):
 if "optim" in p.name or "_ema_" in p.name: continue
 try: d=torch.load(p,map_location="cpu",weights_only=True)
 except Exception: continue
 if isinstance(d,dict) and any(k.startswith("rwkv_modules.0.blocks.0.time_mixer.W_r") for k in d):
  print(p, sum(v.numel() for v in d.values() if isinstance(v,torch.Tensor)))
