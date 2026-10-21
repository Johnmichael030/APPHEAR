#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$python=Join-Path $run 'runtime/venv/Scripts/python.exe'
& $python -B (Join-Path $PSScriptRoot 'inspect_datasets.py') --run-root $run
if($LASTEXITCODE -ne 0){throw 'Metadata validation failed'}

