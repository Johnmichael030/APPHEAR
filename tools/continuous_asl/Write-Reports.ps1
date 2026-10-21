#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
& (Join-Path $run 'runtime/venv/Scripts/python.exe') -B (Join-Path $PSScriptRoot 'generate_reports.py') --run-root $run
if($LASTEXITCODE -ne 0){throw 'REPORT_GENERATION_FAILED'}
