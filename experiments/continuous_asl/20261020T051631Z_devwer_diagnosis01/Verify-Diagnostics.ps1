$ErrorActionPreference='Stop'
$repo=(Get-Item $PSScriptRoot).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
& $python -B (Join-Path $PSScriptRoot 'verify_outputs.py')
if($LASTEXITCODE -ne 0){throw 'DIAGNOSTIC_READ_ONLY_VERIFICATION_FAILED'}
