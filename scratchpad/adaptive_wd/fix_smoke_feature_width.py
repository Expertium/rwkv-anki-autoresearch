from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    if args.smoke_from_scratch:
        import sys
        sys.path.insert(0,str(ROOT))
        import torch,tomli
''','''    if args.smoke_from_scratch:
        import torch,tomli
''')
s=s.replace('''    if args.smoke_from_scratch:
        import torch,tomli
''','''    if args.smoke_from_scratch:
        import torch,tomli
''')
# use prefix adaptation for 17 wider ID features where initial ckpt is narrower
s=s.replace('''        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
        start_dir=run_dir/"smoke_start"; start_dir.mkdir(parents=True,exist_ok=True)
''','''        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
        # Temporary random smoke checkpoint only: live source may have ID-feature width 109,
        # while checkpoint without id features has 92. The additional columns are zero initialized.
        sd=model.state_dict()
        key="features2card.0.weight"
        if sd[key].shape[1]==109:
            sd[key][:,17:]=0.0
        model.load_state_dict(sd,strict=True)
        start_dir=run_dir/"smoke_start"; start_dir.mkdir(parents=True,exist_ok=True)
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('make scratch smoke checkpoint match active 109-feature runtime')
