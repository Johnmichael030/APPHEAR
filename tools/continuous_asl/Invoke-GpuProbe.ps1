#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot, [string]$PythonPath, [switch]$RequireCuda)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run = Get-ExperimentRoot $RunRoot
if (!$PythonPath) { $PythonPath = Join-Path $run 'runtime/venv/Scripts/python.exe' }
$output = Assert-IsolatedOutput (Join-Path $run 'logs/gpu_probe.json') $run
$options = @('-B',(Join-Path $PSScriptRoot 'gpu_probe.py'),'--output',$output)
if ($RequireCuda) { $options += '--require-cuda' }
& $PythonPath @options
$code = $LASTEXITCODE
if ($code -ne 0) { throw "GPU probe did not pass (exit $code); see $output" }

