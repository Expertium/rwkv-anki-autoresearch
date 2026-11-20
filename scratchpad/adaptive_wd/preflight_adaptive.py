import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scratchpad'/'adaptive_wd'))
import lmdb
from rwkv.train_rwkv import get_groups
import adaptive_train as a
train=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'; val=r'F:/rwkv_lmdb/test_db_5k_id5'; label=r'F:/rwkv_lmdb/label_filter_db_id_e2s'
for p in (train,val,label): assert Path(p).exists(), f'missing {p}'
g=get_groups(train,400_000_000_000,16384,list(range(1,5001)))
assert len(g)==43354, len(g)
print('TRAIN_GROUPS_16384',len(g))
u,x=a.choose_screen_users(val)
assert len(u)==750 and len(set(u))==750 and all(5001<=v<=7500 for v in u)
print('SCREEN_ROSTER',len(u),'excluded_val_multibatch',len(x),'first_last',u[0],u[-1])
