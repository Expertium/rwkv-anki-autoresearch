from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if args.smoke_from_scratch:
        import torch,tomli
        from argparse import Namespace
        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
''','''    if args.smoke_from_scratch:
        import sys
        sys.path.insert(0,str(ROOT))
        import torch,tomli
        from argparse import Namespace
        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('fixed import path for script-mode smoke')
