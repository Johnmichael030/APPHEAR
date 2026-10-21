#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[ValidatePattern('^[A-Za-z0-9_]+$')][string]$CaseTag='hazards_02')
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$python=Join-Path $run 'runtime/venv/Scripts/python.exe'
& $python -B (Join-Path $PSScriptRoot 'regression_cases.py') --run-root $run --case-tag $CaseTag
if($LASTEXITCODE -ne 0){throw 'HAZARD_REGRESSION_FAILED'}
