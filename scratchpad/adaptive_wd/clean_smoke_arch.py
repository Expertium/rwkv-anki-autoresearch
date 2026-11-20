from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        # Isolated plumbing smoke must match the current architecture used by the actual run.
        from rwkv.train_rwkv import _maybe_set_aniki_id_features
        _maybe_set_aniki_id_features()
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
''','''        # Isolated plumbing smoke must match the current architecture; feature flags are set
        # before trainer imports this module in a fresh subprocess.
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('remove ineffective late feature hook; smoke feature env is set by runner before process startup')
