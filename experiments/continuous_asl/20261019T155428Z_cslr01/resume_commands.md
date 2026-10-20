```powershell
$repo = 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite'
$run = 'D:/apphearv3claude/AppHear-claude-s2-r3-eval-suite/AppHear-claude-s2-r3-eval-suite/experiments/continuous_asl/20261019T155428Z_cslr01'
Set-Location -LiteralPath $repo
& "$repo/tools/continuous_asl/Verify-Preservation.ps1" -RunRoot $run
# Already early-stopped: this safely reports completion without more epochs.
& "$repo/tools/continuous_asl/Invoke-Training.ps1" -RunRoot $run -Resume
```
```powershell
# Optional fresh reproduction; do not rerun completed preprocessing/export into this run.
# Reserve at least 16 GiB free for another isolated environment and retain all old runs.
$next = & "$repo/tools/continuous_asl/New-Experiment.ps1" -TemplateRunRoot $run
& "$repo/tools/continuous_asl/Initialize-Environment.ps1" -RunRoot $next -CorePackages -PerceptionPackages -ExportPackages
& "$repo/tools/continuous_asl/Invoke-GpuProbe.ps1" -RunRoot $next -RequireCuda
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -DryRun
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -Ids ncslgr_video_index -Extract
& "$repo/tools/continuous_asl/Invoke-MetadataValidation.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -ManifestPath "$next/subset_acquisition_plan.json" -MaxTotalBytes 83886080 -DryRun
& "$repo/tools/continuous_asl/Invoke-Download.ps1" -RunRoot $next -ManifestPath "$next/subset_acquisition_plan.json" -MaxTotalBytes 83886080
& "$repo/tools/continuous_asl/Invoke-Preprocessing.ps1" -RunRoot $next -Tag full
& "$repo/tools/continuous_asl/Invoke-Baseline.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Training.ps1" -RunRoot $next
& "$repo/tools/continuous_asl/Invoke-Evaluation.ps1" -RunRoot $next
```

This reproduces the diagnostic protocol, not guaranteed identical CUDA CTC bits. Endpoints may change; checksum/safety failures must stop and retain source files. No command can finish the blocked signer/boundary/rejection/device gates without new authorized evidence. Do not start more training on unchanged tiny data as the recommended next research action.
