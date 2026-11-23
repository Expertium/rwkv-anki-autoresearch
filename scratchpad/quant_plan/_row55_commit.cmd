@echo off
cd /d C:\Users\Andrew\rwkv-anki-autoresearch
del /f /q .git\index.lock 2>nul
git add scratchpad/quant_plan/corruption/_build_row55_bundle.py
git add scratchpad/quant_plan/bucket_tax_row55.json
git add scratchpad/quant_plan/_row55_bucket_notrun.cmd scratchpad/quant_plan/_row55_clean_bundle.cmd
git add scratchpad/quant_plan/_row55_deploy_evidence.cmd scratchpad/quant_plan/_row55_probe.cmd scratchpad/quant_plan/_export_row55_weights.py
git add scratchpad/row55_l2i8s4e1c_bundle/.gitignore scratchpad/row55_l2i8s4e1c_bundle/README.md scratchpad/row55_l2i8s4e1c_bundle/ANSWERS.md scratchpad/row55_l2i8s4e1c_bundle/verification.md scratchpad/row55_l2i8s4e1c_bundle/verify.cmd scratchpad/row55_l2i8s4e1c_bundle/MANIFEST.md
git add scratchpad/row17_l2i8s4_bundle/verification.md
git add AGENTS.md
git add scratchpad/quant_plan/autorun/QUEUE.md
git add optimization/GPU_RUN_BOARD.md
echo --- staged ---
git diff --cached --name-status
git commit -m "iter 55 bundle (l2i8s4e1c): generated README/ANSWERS/verification/verify.cmd/MANIFEST, answers the other team's six questions from iter 55's own deploy numbers, drops the 23 MB long trace (43.7 MB -> 17.4 MB), records the GPU stop"
echo --- done ---
git log --oneline -1
