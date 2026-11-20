import sys,contextlib,io
sys.path.insert(0,'.')
from rwkv.train_rwkv import get_groups
p=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'
for maxlen in [16384,65536]:
 b=io.StringIO()
 with contextlib.redirect_stdout(b): g=get_groups(p,400_000_000_000,maxlen,list(range(1,5001)))
 users={k[0] for group in g for k in group}
 print('maxlen',maxlen,'groups',len(g),'users',len(users),'missing',5000-len(users))
