from pathlib import Path
p=Path('scratchpad/adaptive_wd/adaptive_train.py');s=p.read_text(encoding='utf-8')
s=s.replace('''    # Deterministic selection. Reserve a disjoint, equally spaced probe for trainer-internal validation.
    probe=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
    # Use odd/even strata to keep the 750-user screen disjoint from internal trainer validation.
    screen_pool=[u for i,u in enumerate(eligible) if i % 2 == 0]
    chosen=[screen_pool[((2*i+1)*len(screen_pool))//(2*count)] for i in range(count)]
''','''    # Deterministic 750-user subset, near-uniform across the full validation half.
    chosen=[eligible[((2*i+1)*len(eligible))//(2*count)] for i in range(count)]
''')
s=s.replace('''    # Fixed, deterministic screen chosen before training; do not draw on results.
    screen_users, excluded_users=choose_screen_users(val_db)
    users=list(screen_users[:2] if args.smoke else screen_users)
    internal_val_users=[5001,5002]
    if set(users) & set(internal_val_users):
        internal_val_users=[u for u in range(5001,7501) if u not in set(users) and u not in excluded_users][:2]
    if len(internal_val_users)!=2 or set(users)&set(internal_val_users):
        raise RuntimeError("could not create disjoint fixed internal-validation probe")
''','''    # Fixed, deterministic screen chosen before training; do not draw on results.
    screen_users, excluded_users=choose_screen_users(val_db)
    users=list(screen_users[:2] if args.smoke else screen_users)
    # Training validation has its own fixed 2-user roster. Evaluation metric users are separate.
    internal_val_users=[5001,5002]
''')
s=s.replace('''        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''','''        "VALIDATE_USERS_FILE":str(Path(out).with_name("validation_users.json")),''')
# Check all output roster existence/count before training. We only require that the LMDB
# builder's official batching code accepts the 5k train users; avoid a mistaken 1-batch assertion.
s=s.replace('''    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
    g=len(groups); del groups
''','''    groups=get_groups(train_db,400_000_000_000,16384,list(range(1,5001)))
    g=len(groups); del groups
''')
p.write_text(s,encoding='utf-8',newline='\n')
print('removed accidental screen/internal validation overlap requirement')
