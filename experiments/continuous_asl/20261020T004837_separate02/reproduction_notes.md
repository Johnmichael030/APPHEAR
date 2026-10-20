# Exact executed commands and safe continuation

All commands used working directory `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite` and interpreter `D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe`. No dependency reinstall, global Python or driver change occurred. Primary commands:

```powershell
$repo='D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite'
$run='D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261020T004837_separate02'
Set-Location -LiteralPath $repo
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Recover
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainA
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase BaselineA
& "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run -DryRun
& "$repo/tools/asl_research_v2/Invoke-How2Sign.ps1" -RunRoot $run
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase How2Verify
& 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe' -B "$repo/tools/asl_research_v2/prepare_b_features.py" --run-root $run --tag full
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase TrainB
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Transfer
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase WarmStart
& "$repo/tools/asl_research_v2/Invoke-Phase.ps1" -RunRoot $run -Phase Compare
& 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe' -B "$repo/tools/asl_research_v2/warmstart_repeat.py" --run-root $run
```

These commands are execution history, not an instruction to overwrite completed outputs. Fresh-output guards refuse existing stages. `reproduction_commands.ps1` provides safe continuation: completed-run resumes verify the latest checkpoint and start no extra epochs; acquisition reuses a verified complete archive without network download. For a new replication, create a new named run via the existing New-Experiment.ps1 initializer, stage the verified TRAIN/VAL sources and transfer fixtures there, keep its source/feature/config hashes, and reuse existing verified media instead of downloading it again. Do not choose a new seed/checkpoint based on TEST data. Restore the exact missing phase7 source artifact from an authoritative archive to its recorded path/hash; a similarly named legacy model is not a substitute.

Broader tests used regression_sandbox.py on an independent copy with external writes/subprocess/network denied. Native Windows long-path prefix handling and pytest's default null log target needed harness corrections; retained earlier attempts are not model-regression evidence. Full tests still have missing MediaPipe and historical fixture/API/snapshot failures. Use a separately validated regression environment if resolving dependencies; do not modify the reused completed-run environment. Exact error messages are in logs/full_regression_final.txt and XML. No command can manufacture missing consented negatives, signer/timing XML or Android device evidence.
