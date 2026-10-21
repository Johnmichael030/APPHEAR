#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
& (Join-Path $run 'runtime/venv/Scripts/python.exe') -B (Join-Path $PSScriptRoot 'evaluate_export.py') --run-root $run
if($LASTEXITCODE -ne 0){throw 'EVALUATION_OR_EXPORT_FAILED; successful artifacts retained'}
