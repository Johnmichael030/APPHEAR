#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$baseline=Get-Content (Join-Path $run 'preservation_baseline.json') -Raw | ConvertFrom-Json -AsHashtable
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if($baseline.repository_root -ne $repo){throw 'BASELINE_REPOSITORY_MISMATCH'}
$mismatches=@();$metadataChanges=@();$bytes=0L
foreach($record in $baseline.files){
    $path=Join-Path $repo $record.path
    if(!(Test-Path -LiteralPath $path -PathType Leaf)){$mismatches+=@{path=$record.path;status='MISSING'};continue}
    $item=Get-Item -LiteralPath $path
    $hash=Get-FileSha256 $path
    if($hash -ne $record.sha256 -or $item.Length -ne $record.bytes){$mismatches+=@{path=$record.path;status='CHANGED';observed_sha256=$hash}}
    if($item.LastWriteTimeUtc.Ticks -ne ([DateTimeOffset]$record.last_write_utc).UtcTicks){$metadataChanges+=@{path=$record.path;observed_last_write_utc=$item.LastWriteTimeUtc.ToString('o')}}
    $bytes+=$item.Length
}
$result=@{status=if($mismatches.Count -or $metadataChanges.Count){'FAILED'}else{'PASS'};verified_utc=[DateTime]::UtcNow.ToString('o');original_files_checked=$baseline.files.Count;original_bytes_checked=$bytes;content_mismatches=$mismatches;last_write_changes=$metadataChanges;baseline_sha256=Get-FileSha256 (Join-Path $run 'preservation_baseline.json');git_available=$false;production_model_assets_avatars_manifests='included in all-original-file verification'}
Write-ExperimentJson $result (Join-Path $run 'preservation_verification.json') $run
$result | ConvertTo-Json -Depth 6
if($result.status -ne 'PASS'){throw 'ORIGINAL_FILE_PRESERVATION_FAILED'}
