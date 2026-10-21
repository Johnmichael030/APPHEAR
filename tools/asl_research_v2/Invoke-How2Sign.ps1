#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[switch]$DryRun,[string]$ManifestPath)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
Import-Module (Join-Path $PSScriptRoot 'DownloadV2.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
if(!$ManifestPath){$ManifestPath=Join-Path $run 'how2sign_currentpage_plan.json'}
$plan=@(Get-DownloadPlan $ManifestPath $run @() 2147483648)
if($DryRun){$plan|Select-Object id,target,max_bytes|ConvertTo-Json;return}
$records=@()
try{foreach($entry in $plan){$result=Invoke-PublicDownload $entry $run;$records+=[PSCustomObject]@{id=$entry.id;source_url=$entry.url;path=$entry.target;status=$result.status;bytes=$result.bytes;sha256=$result.sha256};$result|ConvertTo-Json -Compress}}
finally{if($records.Count){$records|Export-Csv (Join-Path $run 'how2sign_download_manifest.csv') -Append -NoTypeInformation -Encoding utf8NoBOM}}
