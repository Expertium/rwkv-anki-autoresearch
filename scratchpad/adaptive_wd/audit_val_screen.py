import lmdb,json
p=r'F:/rwkv_lmdb/test_db_5k_id5'
e=lmdb.open(p,map_size=250_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
t=e.begin(); eligible=[]; missing=[]; multi=[]
for u in range(5001,7501):
 raw=t.get(f'{u}_batches'.encode())
 if raw is None: missing.append(u); continue
 n=len(json.loads(raw))
 if n!=1: multi.append((u,n))
 else: eligible.append(u)
print('VAL users',2500,'single_batch',len(eligible),'missing',len(missing),'multibatch',len(multi),'multi',multi)
e.close()
