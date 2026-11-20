import lmdb,json
p=r'F:/rwkv_lmdb/test_db_5k_id5'
env=lmdb.open(p,map_size=250_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
with env.begin() as t:
 out=[]
 for u in range(5001,5751):
  b=t.get(f'{u}_batches'.encode())
  if b is not None:
   q=json.loads(b); out.append((u,len(q),sum(int(x[2]) for x in q)))
print('users',len(out),'multibatch',sum(n>1 for _,n,_ in out),'missing',750-len(out),'examples',[x for x in out if x[1]>1][:5])
env.close()
