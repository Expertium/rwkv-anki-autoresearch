from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    ap.add_argument("--start-optim",default="")
    ap.add_argument("--run-dir",required=True)
''','''    ap.add_argument("--start-optim",default="")
    ap.add_argument("--smoke-from-scratch",action="store_true",help="create an isolated random start checkpoint for plumbing smoke only")
    ap.add_argument("--run-dir",required=True)
''')
s=s.replace('''    if not args.smoke and not args.start_model: ap.error("full run requires --start-model")
    if args.smoke and not args.start_model: ap.error("smoke needs the real champion checkpoint")
    start_model=Path(args.start_model).resolve()
    if not start_model.exists(): ap.error(f"missing start checkpoint {start_model}")
    start_optim=Path(args.start_optim).resolve() if args.start_optim else start_model.with_name(start_model.stem+"_optim.pth")
    if not start_optim.exists(): ap.error(f"missing optimizer pair {start_optim}")
''','''    if args.smoke_from_scratch and not args.smoke: ap.error("--smoke-from-scratch is smoke-only")
    if not args.start_model and not args.smoke_from_scratch: ap.error("--start-model required except for --smoke-from-scratch")
    if args.smoke_from_scratch:
        import torch,tomli
        from argparse import Namespace
        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from rwkv.model.srs_model import SrsRWKV
        from rwkv.train_rwkv import get_optimizer
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
        start_dir=run_dir/"smoke_start"; start_dir.mkdir(parents=True,exist_ok=True)
        start_model=start_dir/"start.pth"
        torch.save(model.state_dict(),start_model)
        raw=tomli.loads(train_cfg.read_text(encoding="utf-8"))
        raw["PEAK_LR"]=0.001
        cfgobj=argparse.Namespace(**raw)
        optimizer=get_optimizer(cfgobj,model)
        start_optim=start_dir/"start_optim.pth"
        torch.save(optimizer.state_dict(),start_optim)
        del model,optimizer
    else:
        start_model=Path(args.start_model).resolve()
        if not start_model.exists(): ap.error(f"missing start checkpoint {start_model}")
        start_optim=Path(args.start_optim).resolve() if args.start_optim else start_model.with_name(start_model.stem+"_optim.pth")
        if not start_optim.exists(): ap.error(f"missing optimizer pair {start_optim}")
''')
s=s.replace('''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":65536,
''','''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":65536,"PEAK_LR":0.001,
''')
s=s.replace('''        "RWKV_WEIGHT_DECAY":str(wd),"RWKV_WEIGHT_DECAY_CMIX":str(wd),
''','''        "RWKV_WEIGHT_DECAY":str(wd),"RWKV_WEIGHT_DECAY_CMIX":str(wd),
''')
s=s.replace('''    if not 1<=args.max_attempts<=30: ap.error("--max-attempts must be in 1..30")
''','''    if not 1<=args.max_attempts<=30: ap.error("--max-attempts must be in 1..30")
    if not args.smoke and not args.qat_env_json: ap.error("full adaptive procedure requires --qat-env-json for the final QAT epoch")
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('added isolated scratch-start smoke and base LR 1e-3')
