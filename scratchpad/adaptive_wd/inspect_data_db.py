from pathlib import Path
import lmdb,json
for name in ['train_db_5k_h1_id5','test_db_5k_id5']:
 p=Path('F:/rwkv_lmdb')/name
 if not p.exists(): print(name,'missing'); continue
 try:
  e=lmdb.open(str(p),map_size=400_000_000_000,readonly=True,lock=False,readahead=False,max_readers=512)
  with e.begin() as t:
   rows=[]
   for u in [1,2,100,500,1000,2500,5000,5001,5626,5750,7500,10000]:
    b=t.get(f'{u}_batches'.encode())
    if b is not None: rows.append((u,len(json.loads(b))))
  print(name,'ok',rows)
  e.close()
 except Exception as ex: print(name,type(ex).__name__,str(ex)[:300])
