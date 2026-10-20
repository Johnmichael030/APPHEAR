$ErrorActionPreference='Stop'
$repo=(Get-Item $PSScriptRoot).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
$diagnostic=Join-Path $repo 'experiments/continuous_asl/20261020T051631Z_devwer_diagnosis01'
& $python -B (Join-Path $diagnostic 'verify_outputs.py')
if($LASTEXITCODE -ne 0){throw 'DIAGNOSTIC_PRESERVATION_VERIFICATION_FAILED'}
& $python -B (Join-Path $PSScriptRoot 'verify_pilot.py')
if($LASTEXITCODE -ne 0){throw 'PILOT_VERIFICATION_FAILED'}
