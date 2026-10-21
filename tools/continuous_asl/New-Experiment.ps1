#Requires -Version 7.0
param([Parameter(Mandatory)][string]$TemplateRunRoot,[ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunId=([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'_cslr01'),[switch]$DryRun)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$template=Get-ExperimentRoot $TemplateRunRoot
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$parent=Join-Path $repo 'experiments/continuous_asl'
$destination=Assert-IsolatedOutput (Join-Path $parent $RunId) $parent
if(Test-Path -LiteralPath $destination){throw 'RUN_EXISTS_NO_OVERWRITE'}
$paths=@(& rg --files --hidden --no-ignore -g '!**/experiments/**' -g '!**/.git/**' $repo)
if($LASTEXITCODE -ne 0){throw 'INVENTORY_FAILED'}
if($DryRun){@{destination=$destination;protected_file_count=$paths.Count;network=$false;writes=$false;new_environment_requires_free_bytes=17179869184L}|ConvertTo-Json;return}
$files=@(foreach($path in ($paths | Sort-Object)){
    $item=Get-Item -LiteralPath $path
    if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'REPARSE_SOURCE_REQUIRES_REVIEW'}
    @{path=[IO.Path]::GetRelativePath($repo,$path).Replace('\','/');bytes=$item.Length;last_write_utc=$item.LastWriteTimeUtc.ToString('o');sha256=Get-FileSha256 $path}
})
$null=New-Item -ItemType Directory -Path $destination
Write-ExperimentJson @{schema='apphear.continuous_asl.preservation.v1';created_utc=[DateTime]::UtcNow.ToString('o');repository_root=$repo;git_available_for_this_copy=(Test-Path (Join-Path $repo '.git'));original_file_count=$files.Count;files=$files;excluded_scope='Prior experiments are protected by fresh destinations, never mutated'} (Join-Path $destination 'preservation_baseline.json') $destination
foreach($name in @('acquisition_plan.json','subset_acquisition_plan.json','sample_acquisition_plan.json','training_config.yaml')){
    Copy-Item -LiteralPath (Join-Path $template $name) -Destination (Join-Path $destination $name)
}
foreach($dir in @('logs','checkpoints','cache','runtime','regressions','data/raw','data/extracted','data/processed','data/derived')){
    $null=New-Item -ItemType Directory -Path (Join-Path $destination $dir)
}
Write-Output $destination
