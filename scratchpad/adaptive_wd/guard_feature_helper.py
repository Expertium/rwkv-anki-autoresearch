from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        from rwkv.train_rwkv import _maybe_set_aniki_id_features
        _maybe_set_aniki_id_features()
''','''        import rwkv.train_rwkv as _tr
        if hasattr(_tr, "_maybe_set_aniki_id_features"):
            _tr._maybe_set_aniki_id_features()
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('guard feature helper for current trainer API')
