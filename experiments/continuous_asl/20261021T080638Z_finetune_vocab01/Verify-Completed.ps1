$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$pythonPath = Join-Path $projectRoot 'experiments\continuous_asl\20261019T155428Z_cslr01\runtime\venv\Scripts\python.exe'
& $pythonPath -B (Join-Path $PSScriptRoot 'verify_completed.py')
if ($LASTEXITCODE -ne 0) { throw 'Completed fine-tuning artifact verification failed' }
