from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''def make_train_config(base, out, trial_dir, load_folder, load_name, steps,
                      step_offset, train_db, val_db, users):''','''def make_train_config(base, out, trial_dir, load_folder, load_name, load_optim_path, steps,
                      step_offset, train_db, val_db, users):''')
s=s.replace('''        "LOAD_MODEL_NAME":load_name,"SAVE_MODEL_FOLDER":str(trial_dir),''','''        "LOAD_MODEL_NAME":load_name,"LOAD_OPTIM_PATH":str(load_optim_path),
        "SAVE_MODEL_FOLDER":str(trial_dir),''')
s=s.replace('''                          incumbent["model"].stem,(0.04 if args.smoke else epoch_target),''','''                          incumbent["model"].stem,incumbent["optim"],
                          (0.04 if args.smoke else epoch_target),''')
s=s.replace('''                          incumbent["model"].stem,1,1,train_db,val_db,users)''','''                          incumbent["model"].stem,incumbent["optim"],1,1,train_db,val_db,users)''')
p.write_text(s,encoding='utf-8',newline='\n')
print('added explicit optimizer checkpoint path support')
