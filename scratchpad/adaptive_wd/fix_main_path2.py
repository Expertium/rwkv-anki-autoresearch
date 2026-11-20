from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''        import torch,tomli
        from argparse import Namespace
''','''        import torch,tomli
        from argparse import Namespace
''')
# ensure main inserts repo root before local trainer imports and config modules
s=s.replace('''    args=ap.parse_args()
    if not 1<=args.max_attempts<=30:''','''    args=ap.parse_args()
    sys.path.insert(0,str(ROOT))
    if not 1<=args.max_attempts<=30:''')
s=s.replace('''    sys.path.insert(0,str(ROOT))
    from rwkv.train_rwkv import get_groups
''','''    from rwkv.train_rwkv import get_groups
''')
# Add timeout-safe checked command logging diagnostics to wrapper? no
p.write_text(s,encoding='utf-8',newline='\n')
print('ensured main imports resolve from repository root')
