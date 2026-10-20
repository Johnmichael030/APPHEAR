param([ValidateSet('Status','Validate')][string]$Phase='Status')
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$py=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
if($Phase -eq 'Status'){ & (Join-Path $run 'Invoke-Research.ps1') -Phase Status; return }
& $py -B (Join-Path $run 'tools/verify_completed.py') --artifacts
if($LASTEXITCODE -ne 0){throw "COMPLETED_VALIDATION_FAILED_$LASTEXITCODE"}
