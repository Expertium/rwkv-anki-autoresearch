import torch,sys
sys.path.insert(0,'.')
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
m=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
for f in ['pretrain/rwkv/opt_qat46/rwkv_iter46_248.pth','pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth']:
 d=torch.load(f,map_location='cpu',weights_only=True)
 try: m.load_state_dict(d,strict=True); print(f,'STRICT_OK',sum(v.numel() for v in d.values() if isinstance(v,torch.Tensor)))
 except Exception as e: print(f,'FAIL',str(e)[:300])
