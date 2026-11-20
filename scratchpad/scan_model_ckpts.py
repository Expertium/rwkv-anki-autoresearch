from pathlib import Path
import torch
root=Path("pretrain/rwkv")
for p in root.rglob("*.pth"):
 if "optim" in p.name or "_ema_" in p.name: continue
 try: d=torch.load(p,map_location="cpu",weights_only=True)
 except Exception: continue
 if isinstance(d,dict):
  n=sum(v.numel() for v in d.values() if isinstance(v,torch.Tensor))
  if n in (332922,333178,192800,808762): print(n,p)
