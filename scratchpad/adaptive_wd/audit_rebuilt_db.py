import lmdb,json
p=r'F:/rwkv_lmdb/train_db_5k_h1_id5_adaptive'
e=lmdb.open(p,map_size=400_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
t=e.begin(); missing=[]; multi=[]; n=0
for u in range(1,5001):
 r=t.get(f'{u}_batches'.encode())
 if r is None: missing.append(u); continue
 b=json.loads(r); n+=1
 if len(b)!=1: multi.append((u,len(b)))
print('users_with_batch_keys',n,'missing',len(missing),'multi_batch',len(multi),'sample_missing',missing[:10],'sample_multi',multi[:10])
e.close()
