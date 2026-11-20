from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if len(eligible)<count:
        raise RuntimeError(f"Need {count} one-batch VAL users; found {len(eligible)}")
    # Deterministic, near-uniform selection across the full VAL half.
    chosen=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
''','''    if len(eligible)<count:
        raise RuntimeError(f"Need {count} one-batch VAL users; found {len(eligible)}")
    # Deterministic selection. Reserve a disjoint, equally spaced probe for trainer-internal validation.
    probe=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
    # Use odd/even strata to keep the 750-user screen disjoint from internal trainer validation.
    screen_pool=[u for i,u in enumerate(eligible) if i % 2 == 0]
    chosen=[screen_pool[((2*i+1)*len(screen_pool))//(2*count)] for i in range(count)]
''')
s=s.replace('''    users=list(screen_users[:2] if args.smoke else screen_users)
''','''    users=list(screen_users[:2] if args.smoke else screen_users)
    internal_val_users=[5001,5002]
    if set(users) & set(internal_val_users):
        internal_val_users=[u for u in range(5001,7501) if u not in set(users) and u not in excluded_users][:2]
    if len(internal_val_users)!=2 or set(users)&set(internal_val_users):
        raise RuntimeError("could not create disjoint fixed internal-validation probe")
''')
s=s.replace('''        internal_val_users=[5001,5002]
        make_train_config''','''        make_train_config''')
s=s.replace('''        trained_steps=int(0.04*g) if args.smoke else epoch_target*g
        step=trained_steps
''','''        trained_steps=max(1,int(0.04*g)) if args.smoke else epoch_target*g
        step=(1-1)+trained_steps
''')
s=s.replace('''            wd_i+=1
            if wd_i>=len(GRID): rec["stop"]="all WD values exhausted"
''','''            wd_i=next_weight_decay_index(wd_i)
            if wd_i==len(GRID)-1 and wd==GRID[-1]: rec["stop"]="highest WD failed"
''')
s=s.replace('''        if not win and wd_i>=len(GRID): break
''','''        if not win and rec.get("stop")=="highest WD failed": break
''')
s=s.replace('''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":65536,"PEAK_LR":0.001,
''','''        "NUM_FETCH_PROCESSES":1,"MAX_TRAIN_GLOBAL_LEN":16384,"PEAK_LR":0.001,
''')
s=s.replace('''    groups=get_groups(train_db,400_000_000_000,65536,list(range(1,5001)))
''','''    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
''')
s=s.replace('''        qcp=fd/f"adaptive_{g}.pth"; qop=fd/f"adaptive_optim_{g}.pth"
''','''        qcp=fd/f"adaptive_{g}.pth"; qop=fd/f"adaptive_optim_{g}.pth"
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('patched adaptive decisions, 16k groups, disjoint eval split')
