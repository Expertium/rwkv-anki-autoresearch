from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from rwkv.train_rwkv import get_groups
p=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'
g=get_groups(p,400_000_000_000,16384,list(range(1,5001)))
print('train_groups_at_16384',len(g))
assert len(g)>0
