from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from rwkv.model.srs_model import SrsRWKV
        from rwkv.train_rwkv import get_optimizer
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
''','''        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from rwkv.model.srs_model import SrsRWKV
        from rwkv.train_rwkv import get_optimizer
        # Isolated plumbing smoke must match the current architecture used by the actual run.
        from rwkv.train_rwkv import _maybe_set_aniki_id_features
        _maybe_set_aniki_id_features()
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('enable same runtime feature layout in scratch-start smoke')
