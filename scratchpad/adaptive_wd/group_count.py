import sys
sys.path.insert(0,'.')
from rwkv.train_rwkv import get_groups
p=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'
for maxlen in [16384,65536]:
 g=get_groups(p,400_000_000_000,maxlen,list(range(1,5001)))
 print('MAX',maxlen,'groups',len(g),'users',len({k[0] for group in g for k in group}),'first',len(g[0]) if g else 0)
