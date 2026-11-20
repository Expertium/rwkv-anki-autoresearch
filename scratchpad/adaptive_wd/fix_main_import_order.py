from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if not args.smoke_from_scratch and not args.start_model: ap.error("--start-model required except for --smoke-from-scratch")
''','''    if not args.smoke_from_scratch and not args.start_model: ap.error("--start-model required except for --smoke-from-scratch")
    sys.path.insert(0,str(ROOT))
''')
s=s.replace('''        sys.path.insert(0,str(ROOT))
        import torch,tomli
''','''        import torch,tomli
''')
s=s.replace('''    if args.smoke_from_scratch and not args.smoke: ap.error("--smoke-from-scratch is smoke-only")
    if not args.start_model and not args.smoke_from_scratch: ap.error("--start-model required except for --smoke-from-scratch")
''','''    if args.smoke_from_scratch and not args.smoke: ap.error("--smoke-from-scratch is smoke-only")
    if not args.start_model and not args.smoke_from_scratch: ap.error("--start-model required except for --smoke-from-scratch")
    sys.path.insert(0,str(ROOT))
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('inserted repo path before adaptive main imports')
