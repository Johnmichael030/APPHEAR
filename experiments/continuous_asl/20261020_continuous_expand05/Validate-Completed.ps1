$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$py=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
& $py -B (Join-Path $run 'tools/verify_completed.py')
if($LASTEXITCODE -ne 0){throw "FINAL_VALIDATION_FAILED_$LASTEXITCODE"}
