import lmdb,json
p=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'
e=lmdb.open(p,map_size=400_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
t=e.begin(); missing=[]; bad=[]
for u in range(1,5001):
 raw=t.get(f'{u}_batches'.encode())
 if raw is None: missing.append(u); continue
 batches=json.loads(raw)
 if len(batches)!=1: bad.append((u,len(batches)))
print('train_users_missing',len(missing),'train_multibatch',len(bad),'examples',bad[:8])
assert not missing and not bad, 'adaptive training LMDB must have exactly one batch per user'
e.close()
