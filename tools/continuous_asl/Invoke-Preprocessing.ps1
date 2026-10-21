#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[ValidateSet('sample','full')][string]$Tag='full',[string[]]$Ids)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$python=Join-Path $run 'runtime/venv/Scripts/python.exe'
$arguments=@('-B',(Join-Path $PSScriptRoot 'prepare_features.py'),'--run-root',$run,'--tag',$Tag)
if($Ids){$arguments+=@('--ids')+$Ids}
& $python @arguments
if($LASTEXITCODE -ne 0){throw "PREPROCESSING_FAILED:$LASTEXITCODE; retained all source and successful outputs"}
