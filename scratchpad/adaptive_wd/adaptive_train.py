import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
GRID = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
SCREEN_START, SCREEN_END, SCREEN_COUNT = 5001, 7500, 750

def choose_screen_users(val_db, count=SCREEN_COUNT):
    import lmdb
    env=lmdb.open(val_db,map_size=250_000_000_000,readonly=True,lock=False,
                  readahead=False,max_readers=512)
    eligible=[]; excluded=[]
    with env.begin() as txn:
        for user in range(SCREEN_START,SCREEN_END+1):
            raw=txn.get(f"{user}_batches".encode())
            if raw is None: excluded.append(user); continue
            if len(json.loads(raw)) != 1: excluded.append(user); continue
            eligible.append(user)
    env.close()
    if len(eligible)<count:
        raise RuntimeError(f"Need {count} one-batch VAL users; found {len(eligible)}")
    # Deterministic 750-user subset, near-uniform across the full validation half.
    chosen=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
    return tuple(chosen), tuple(excluded)


def should_accept(previous, candidate):
    """Strict dual-mode screen rule: both by-user means must go down."""
    return candidate["ahead"] < previous["ahead"] and candidate["imm"] < previous["imm"]

def next_weight_decay_index(current):
    """Move one rung up the fixed WD grid after a rejected epoch; saturate at last rung."""
    return min(current+1,len(GRID)-1)

def resume_schedule(state):
    """Recover the next WD rung and epoch target from persisted adaptive trials."""
    trials = state.get("trials", [])
    if not trials:
        raise ValueError("resume state has no completed trials")
    failed = sum(t.get("decision") == "rollback" for t in trials)
    recorded_failed = state.get("failed_epochs", failed)
    if recorded_failed != failed:
        raise ValueError(f"failed-epoch count mismatch: state={recorded_failed}, trials={failed}")
    last = trials[-1]
    try:
        last_i = GRID.index(float(last["weight_decay"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("resume state has an invalid weight-decay rung") from exc
    if last.get("decision") == "accept":
        wd_i = last_i
    elif last.get("decision") == "rollback":
        wd_i = next_weight_decay_index(last_i)
    else:
        raise ValueError("resume state has an invalid trial decision")
    return wd_i, int(state.get("accepted_epochs", 0)) + 1, failed


def toml_repr(value):
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, str): return json.dumps(value)
    return str(value)


def edit_toml(path, values):
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen, out = set(), []
    for line in lines:
        raw = line.split("#", 1)[0]
        if "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in values:
                if key not in seen:
                    out.append(f"{key} = {toml_repr(values[key])}")
                    seen.add(key)
                # Drop every duplicate occurrence of keys we override.
                continue
        out.append(line)
    for key, value in values.items():
        if key not in seen: out.append(f"{key} = {toml_repr(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run_checked(argv, cwd, logfile, env):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([str(x) for x in argv], cwd=str(cwd), env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=False)
    if p.returncode:
        tail = "\n".join(logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"exit {p.returncode}: {argv}\n{tail}")


def eval_config(base_path, output, model_path, tag, users, val_db, label_db):
    users_file=Path(output).with_name(f"users_{tag}.json")
    users_file.write_text(json.dumps(list(users)),encoding="utf-8")
    values = {
        "MODEL_PATH": str(model_path), "FILE_AHEAD": f"adw_{tag}",
        "FILE_IMM": f"adwP_{tag}", "DATASET_LMDB_PATH": str(val_db),
        "DATASET_LMDB_SIZE": 250_000_000_000,
        "LABEL_FILTER_LMDB_PATH": str(label_db), "LABEL_FILTER_LMDB_SIZE": 40_000_000_000,
        "USER_START": min(users), "USER_END": max(users),
        "USERS_FILE": str(users_file), "NUM_FETCH_PROCESSES": 1, "RAW": False,
    }
    shutil.copy2(base_path, output)
    edit_toml(output, values)


def eval_users(run_dir, base_eval, model, tag, users, val_db, label_db, env, shards=1, require_full=True):
    cfg = Path(run_dir) / f"eval_{tag}.toml"
    eval_config(base_eval, cfg, model, tag, users, val_db, label_db)
    users_path=Path(run_dir)/f"users_{tag}.json"
    users_path.write_text(json.dumps(list(users)),encoding="utf-8")
    run_checked([PY, ROOT / "optimization/eval_sharded.py", "--config", cfg,
                 "--users-file", users_path, "--shards", shards, "--solo-threshold", 0,
                 "--fetch-per-shard", 1, "--threads-per-shard", 7],
                ROOT, Path(run_dir) / f"eval_{tag}.log", env)
    pa, pi = ROOT / "result" / f"adw_{tag}.jsonl", ROOT / "result" / f"adwP_{tag}.jsonl"
    if not pa.exists() or not pi.exists(): raise RuntimeError(f"missing merged eval for {tag}")
    rows=[]
    for p in (pa,pi):
        d={int(x["user"]):x for x in (json.loads(z) for z in p.read_text(encoding="utf-8").splitlines() if z.strip())}
        rows.append(d)
    if not rows[0] or set(rows[0]) != set(rows[1]): raise RuntimeError("empty or unequal mode roster")
    if require_full and set(rows[0]) != set(users):
        missing=sorted(set(users)-set(rows[0]))
        raise RuntimeError(f"screen eval did not score the full fixed roster (expected {len(users)}, scored {len(rows[0])}, missing {missing[:20]})")
    n=len(rows[0]); means=[]
    for d in rows:
        means.append(sum(float(x["metrics"]["LogLoss"]) for x in d.values())/n)
    return {"ahead":means[0],"imm":means[1],"n":n}


def make_train_config(base, out, trial_dir, load_folder, load_name, load_optim_path, steps,
                      step_offset, train_db, val_db, label_db, users):
    try: import tomllib
    except ImportError: import tomli as tomllib
    data=tomllib.loads(Path(base).read_text(encoding="utf-8"))
    values={
        "TRAIN_USERS_START":1,"TRAIN_USERS_END":5000,
        "VALIDATE_USERS_START":min(users),"VALIDATE_USERS_END":max(users),
        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),
        "TRAIN_DATASET_LMDB_PATH":str(train_db),"TRAIN_DATASET_LMDB_SIZE":400_000_000_000,
        "VALIDATE_DATASET_LMDB_PATH":str(val_db),"VALIDATE_DATASET_LMDB_SIZE":250_000_000_000,
        "LABEL_FILTER_LMDB_PATH":str(label_db),"LABEL_FILTER_LMDB_SIZE":40_000_000_000,
        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,"PEAK_LR":0.001,
        "VALIDATE_EVERY": max(1,int(0.02*steps)) if isinstance(steps,float) else 500,
        "TRAIN_MODE":"D","STEP_OFFSET":step_offset,"WARMUP_STEPS":1,
        "EPOCHS":steps,"LOAD_MODEL":True,"LOAD_MODEL_FOLDER":str(load_folder),
        "LOAD_MODEL_NAME":load_name,"LOAD_OPTIM_PATH":str(load_optim_path),
        "SAVE_MODEL_FOLDER":str(trial_dir),
        "SAVE_MODEL_PREFIX":"adaptive","USE_WANDB":False,"DEVICE":"cuda","DTYPE":"bfloat16",
    }
    shutil.copy2(base,out); edit_toml(out,values)


def read_validation_log(path):
    text=Path(path).read_text(encoding="utf-8",errors="replace")
    ms=re.findall(r"Mean ahead validation loss:\s*([0-9.eE+-]+).*?imm:\s*([0-9.eE+-]+), validation n:\s*(\d+)",text)
    if not ms: raise RuntimeError("training log has no complete validation summary")
    a,i,n=ms[-1]; a,i,n=float(a),float(i),int(n)
    if not math.isfinite(a) or not math.isfinite(i) or n<1: raise RuntimeError("invalid validation result")
    if "Traceback (most recent call last)" in text or "Exception caught." in text:
        raise RuntimeError("trainer log reports exception/traceback")
    return {"ahead":a,"imm":i,"n":n}


def main():
    ap=argparse.ArgumentParser(description="Adaptive per-epoch WD trials with exact rollback.")
    ap.add_argument("--train-config",default=str(ROOT/"rwkv/train_rwkv_config.toml"))
    ap.add_argument("--eval-config",default=str(ROOT/"rwkv/get_result_config_base5k.toml"))
    ap.add_argument("--train-db",default="F:/rwkv_lmdb/train_db_5k_h1_adaptive92")
    ap.add_argument("--val-db",default="F:/rwkv_lmdb/test_db_5k_adaptive92")
    ap.add_argument("--label-db",default="F:/rwkv_lmdb/label_filter_db_id_e2s")
    ap.add_argument("--start-model",default="")
    ap.add_argument("--start-optim",default="")
    ap.add_argument("--smoke-from-scratch",action="store_true",help="create an isolated random start checkpoint for plumbing smoke only")
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--max-failed-epochs",type=int,default=7)
    ap.add_argument("--max-successful-epochs",type=int,default=10)
    ap.add_argument("--resume-state",default="",help="resume the adaptive schedule from a saved adaptive_state.json")
    ap.add_argument("--fetchers",type=int,default=1)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--qat-env-json",default="")
    ap.add_argument("--full-eval",action="store_true",help="after final QAT, run canonical full 5001-7500 VAL")
    args=ap.parse_args()
    import sys
    sys.path.insert(0,str(ROOT))
    if not 1<=args.max_failed_epochs<=len(GRID): ap.error(f"--max-failed-epochs must be in 1..{len(GRID)}")
    if not 1<=args.max_successful_epochs<=10: ap.error("--max-successful-epochs must be in 1..10")
    if args.smoke and not args.smoke_from_scratch: ap.error("smoke mode must use --smoke-from-scratch")
    if not args.smoke and not args.qat_env_json: ap.error("full adaptive procedure requires --qat-env-json for the final QAT epoch")
    if args.fetchers!=1: ap.error("use exactly one fetch worker")
    run_dir=Path(args.run_dir).resolve(); run_dir.mkdir(parents=True,exist_ok=True)
    train_cfg=Path(args.train_config).resolve(); eval_base=Path(args.eval_config).resolve()
    train_db=args.train_db; val_db=args.val_db; label_db=args.label_db
    # Train/eval TOML overrides must match the CLI-selected database paths.
    for p in (train_db,val_db,label_db):
        if not Path(p.replace("/","\\")).exists(): ap.error(f"missing input path: {p}")
    if args.smoke_from_scratch and not args.smoke: ap.error("--smoke-from-scratch is smoke-only")
    if not args.start_model and not args.smoke_from_scratch and not args.resume_state: ap.error("--start-model required except for smoke or --resume-state")
    sys.path.insert(0,str(ROOT))
    if args.smoke_from_scratch:
        import sys
        import torch,tomli
        from argparse import Namespace
        from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from rwkv.model.srs_model import SrsRWKV
        from rwkv.train_rwkv import get_optimizer
        # Isolated plumbing smoke must match the current architecture used by the actual run.
        import rwkv.train_rwkv as _tr
        if hasattr(_tr, "_maybe_set_aniki_id_features"):
            _tr._maybe_set_aniki_id_features()
        model=SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG).float()
        # Smoke must use the same 92-input layout as the production checkpoint.
        sd=model.state_dict()
        key="features2card.0.weight"
        if sd[key].shape[1] != 92:
            raise RuntimeError(f"92-input smoke required; model has {sd[key].shape[1]} inputs")
        model.load_state_dict(sd,strict=True)
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
    elif args.resume_state:
        resume_path=Path(args.resume_state).resolve()
        if not resume_path.exists(): ap.error(f"missing resume state {resume_path}")
        start_model=None; start_optim=None
    else:
        start_model=Path(args.start_model).resolve()
        if not start_model.exists(): ap.error(f"missing start checkpoint {start_model}")
        start_optim=Path(args.start_optim).resolve() if args.start_optim else start_model.with_name(start_model.stem+"_optim.pth")
        if not start_optim.exists(): ap.error(f"missing optimizer pair {start_optim}")
    env0=dict(os.environ)
    env0["RWKV_ID_FEATURES"]="0"
    env0["RWKV_REAL_CYCLES"]="0"
    env0["RWKV_FINAL_LAYOUT"]="0"
    env0["RWKV_E2S_PUBLISHED"]="0"
    # The champion and adaptive LMDBs use the 92-input layout.
    # Fail early if a caller's process has enabled optional features that change it.
    from rwkv.id_features import input_width
    if input_width() != 92:
        raise RuntimeError(f"adaptive run requires 92 inputs; current process layout is {input_width()}")
    import sys
    from rwkv.train_rwkv import get_groups
    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
    g=len(groups); del groups
    if args.smoke:
        g=min(g,8)
    if g<1: raise RuntimeError("training DB has no groups")
    # Fixed, deterministic screen chosen before training; do not draw on results.
    screen_users, excluded_users=choose_screen_users(val_db)
    users=list(screen_users[:1] if args.smoke else screen_users)
    # Both internal validation and adaptive screen require the trainer's explicit-user file.
    # Training validation has its own fixed 2-user roster, disjoint from evaluation roster.
    internal_val_users=[u for u in range(5001,7501) if u not in set(users) and u not in excluded_users][:2]
    if len(internal_val_users)<2: raise RuntimeError("could not select two internal validation users")
    validation_users_path=run_dir/"validation_users.json"
    validation_users_path.write_text(json.dumps(internal_val_users),encoding="utf-8")
    # Runtime output names are unique; eval_sharded refuses to overwrite any previous result.
    import uuid
    result_tag=run_dir.name+"_"+uuid.uuid4().hex[:8]
    if args.resume_state:
        resume_path=Path(args.resume_state).resolve()
        state=json.loads(resume_path.read_text(encoding="utf-8"))
        users=list(state["users"])
        excluded_users=tuple(state.get("screen_excluded_multibatch_or_missing", excluded_users))
        baseline=state["baseline"]
        stored=state["incumbent"]
        incumbent={"model":Path(stored["model"]).resolve(),"optim":Path(stored["optim"]).resolve(),
                   "score":stored["score"]}
        if not incumbent["model"].exists() or not incumbent["optim"].exists():
            ap.error("saved incumbent model/optimizer pair is missing")
        if users != list(screen_users): ap.error("saved screen roster differs from the deterministic roster")
        wd_i,epoch_target,failed_epochs=resume_schedule(state)
        if not validation_users_path.exists(): ap.error("resume validation-users file is missing")
        internal_val_users=json.loads(validation_users_path.read_text(encoding="utf-8"))
    else:
        base=eval_users(run_dir,eval_base,start_model,result_tag+"_baseline",users,val_db,label_db,env0,shards=1,
                        require_full=False if args.smoke else True)
        if args.smoke and base["n"] < 1: raise RuntimeError("smoke baseline evaluation produced no scored users")
        state={"grid":list(GRID),"attempted":0,"accepted_epochs":0,"failed_epochs":0,
               "users":users,"screen_excluded_multibatch_or_missing":list(excluded_users),
               "baseline":base,"trials":[]}
        incumbent={"model":start_model,"optim":start_optim,
                   "score":{"ahead":base["ahead"],"imm":base["imm"]}}
        wd_i=0; epoch_target=1; failed_epochs=0
    state["max_failed_epochs"]=args.max_failed_epochs
    state["max_successful_epochs"]=args.max_successful_epochs
    state.pop("max_attempts",None)
    state["failed_epochs"]=failed_epochs
    state["incumbent"]={"model":str(incumbent["model"]),"optim":str(incumbent["optim"]),
                          "score":incumbent["score"],"epochs":state["accepted_epochs"]}
    (run_dir/"adaptive_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    cap=1 if args.smoke else None
    while (args.smoke and state["attempted"] < 1) or (not args.smoke and
           failed_epochs < args.max_failed_epochs and
           state["accepted_epochs"] < args.max_successful_epochs):
        attempt=state["attempted"]+1; wd=GRID[wd_i]
        td=run_dir/f"attempt_{attempt:02d}_epoch_{epoch_target:02d}_wd_{wd:g}"; td.mkdir(parents=True,exist_ok=True)
        cfg=td/"train.toml"
        (td/"validation_users.json").write_text(json.dumps(internal_val_users),encoding="utf-8")
        # Trainer validates only against a fixed tiny roster; primary promotion screen is external.
        make_train_config(train_cfg,cfg,td,incumbent["model"].parent,
                          incumbent["model"].stem,incumbent["optim"],
                          (0.04 if args.smoke else epoch_target),
                          (1 if args.smoke else (epoch_target-1)*g+1),
                          train_db,val_db,label_db,internal_val_users)
        env=dict(env0)
        env.update({"RWKV_WEIGHT_DECAY":str(wd),"RWKV_WEIGHT_DECAY_CMIX":str(wd),
                    "RWKV_WEIGHT_DECAY_HEAD":str(wd),"RWKV_RESUME_SKIP_GROUPS":"1",
                    "RWKV_DECAY_RESUME_SCHED":"1","RWKV_DECAY_SHAPE":"constant",
                    "RWKV_VRAM_CAP_GB":"11"})
        log=td/"train.log"
        run_checked([PY,"-u","-m","rwkv.train_rwkv","--config",cfg],ROOT,log,env)
        tx=log.read_text(encoding="utf-8",errors="replace")
        if "MODEL SAVED." not in tx: raise RuntimeError("trainer did not save checkpoint")
        trainer_val=read_validation_log(log)  # checks trainer's own validation ran; gate uses per-user mean below
        trained_steps=max(50,int(0.04*g)) if args.smoke else epoch_target*g
        step=(1-1)+trained_steps
        cp=td/f"adaptive_{step}.pth"; op=td/f"adaptive_optim_{step}.pth"
        if not cp.exists() or not op.exists(): raise RuntimeError("checkpoint pair missing")
        val_result=eval_users(run_dir,eval_base,cp,result_tag+f"_a{attempt:02d}",users,val_db,label_db,env0,shards=1)
        val={"ahead":val_result["ahead"],"imm":val_result["imm"],"users":val_result["n"]}
        prev=incumbent["score"]
        win=should_accept(prev,val)
        rec={"attempt":attempt,"epoch":epoch_target,"weight_decay":wd,"val":val,"trainer_val":trainer_val,
             "previous":prev,"improved_both":win,"checkpoint":str(cp),"train_log":str(log)}
        state["attempted"]+=1
        if win:
            incumbent={"model":cp,"optim":op,"score":{"ahead":val["ahead"],"imm":val["imm"]}}
            state["accepted_epochs"]+=1
            rec["decision"]="accept"
            epoch_target=state["accepted_epochs"]+1
        else:
            failed_epochs+=1
            state["failed_epochs"]=failed_epochs
            rec["decision"]="rollback"
            wd_i=next_weight_decay_index(wd_i)
            if failed_epochs>=args.max_failed_epochs:
                rec["stop"]="maximum failed epochs reached"
        state["trials"].append(rec)
        state["incumbent"]={"model":str(incumbent["model"]),"optim":str(incumbent["optim"]),
                            "score":incumbent["score"],"epochs":state["accepted_epochs"]}
        state["failed_epochs"]=failed_epochs
        state["max_failed_epochs"]=args.max_failed_epochs
        state["max_successful_epochs"]=args.max_successful_epochs
        (run_dir/"adaptive_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        print(json.dumps({"attempt":attempt,"epoch":epoch_target if not win else epoch_target-1,
                          "wd":wd,"val":val,"decision":rec["decision"]}),flush=True)
        if args.smoke: break
        if rec.get("stop")=="maximum failed epochs reached": break
    if args.smoke:
        print("ADAPTIVE_SMOKE_OK",flush=True); return
    qat={}
    if args.qat_env_json:
        qat=json.loads(Path(args.qat_env_json).read_text(encoding="utf-8"))
        if not isinstance(qat,dict): raise RuntimeError("qat env json must be object")
        fd=run_dir/"final_qat"; fd.mkdir(parents=True,exist_ok=True)
        cfg=fd/"train.toml"
        # BUG FIX (2026-11-15): make_train_config points VALIDATE_USERS_FILE at this directory, and
        # the trainer opens that file at startup, before any step. The attempt loop writes the same
        # file per attempt; the QAT branch did not, so the QAT child died with FileNotFoundError
        # after ~2 minutes and the whole controller exited 1. Write it here too, and pass the SAME
        # roster to make_train_config so VALIDATE_USERS_START/END brackets its contents (passing the
        # 750-user screen roster instead trips the trainer's subset assertion).
        (fd/"validation_users.json").write_text(json.dumps(internal_val_users),encoding="utf-8")
        make_train_config(train_cfg,cfg,fd,incumbent["model"].parent,
                          incumbent["model"].stem,incumbent["optim"],1,1,train_db,val_db,label_db,internal_val_users)
        env=dict(env0); env.update({str(k):str(v) for k,v in qat.items()})
        env.update({"RWKV_WEIGHT_DECAY":"0.01","RWKV_WEIGHT_DECAY_CMIX":"0.01",
                    "RWKV_WEIGHT_DECAY_HEAD":"0.01","RWKV_VRAM_CAP_GB":"11"})
        log=fd/"qat.log"
        run_checked([PY,"-u","-m","rwkv.train_rwkv","--config",cfg],ROOT,log,env)
        tx=log.read_text(encoding="utf-8",errors="replace")
        if "MODEL SAVED." not in tx: raise RuntimeError("QAT final epoch did not save")
        qcp=fd/f"adaptive_{g}.pth"; qop=fd/f"adaptive_optim_{g}.pth"
        if not qcp.exists() or not qop.exists(): raise RuntimeError("QAT model/optim pair missing")
        state["qat_final"]={"model":str(qcp),"optim":str(qop)}
        if args.full_eval:
            full=eval_users(run_dir,eval_base,qcp,"qat_full",list(range(5001,7501)),val_db,label_db,env0,shards=2,require_full=False)
            state["full_eval"]=full
    final_msg=f"ADAPTIVE_DONE attempts={state['attempted']} failed_epochs={state.get('failed_epochs', 0)} accepted_epochs={state['accepted_epochs']} "
    print(final_msg,str(incumbent["model"]),flush=True)

if __name__=="__main__": main()
