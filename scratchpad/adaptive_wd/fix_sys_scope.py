from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    import sys
    sys.path.insert(0,str(ROOT))
    from rwkv.train_rwkv import get_groups
''','''    from rwkv.train_rwkv import get_groups
''')
# sys is now a module-level import; use ROOT once before any package imports.
s=s.replace('''    args=ap.parse_args()
    sys.path.insert(0,str(ROOT))
''','''    args=ap.parse_args()
    import sys
    sys.path.insert(0,str(ROOT))
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('fixed local sys scoping')
