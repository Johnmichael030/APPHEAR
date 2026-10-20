param([ValidateSet('train_dev','test')][string]$Partition='train_dev',[switch]$DryRun)
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$plan=Get-Content (Join-Path $run 'dataset_plan.json') -Raw | ConvertFrom-Json
if($Partition -eq 'test'){
 if(!(Test-Path (Join-Path $run 'model_selection_freeze.json'))){throw 'CUSTOM_HOLDOUT_LOCKED_UNTIL_MODEL_FREEZE'}
 $freeze=Get-Content (Join-Path $run 'model_selection_freeze.json') -Raw | ConvertFrom-Json
 if((Get-FileHash (Join-Path $run 'dataset_plan.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $freeze.plan_sha256){throw 'FROZEN_SPLIT_CHANGED'}
}
$rows=@($plan.rows | Where-Object { !$_.reuse -and (($_.split -eq 'test') -eq ($Partition -eq 'test')) })
$plannedBytes=$rows.Count * [long]$plan.download_max_bytes_per_file
if($plannedBytes -gt [long]$plan.download_max_total_bytes){throw 'DOWNLOAD_TOTAL_CAP'}
if((Get-PSDrive D).Free -lt [long]$plan.disk_headroom_bytes + $plannedBytes){throw 'DISK_HEADROOM'}
$ledger=Join-Path $run ('logs/acquisition_'+$Partition+'_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
$started=[DateTime]::UtcNow; $downloaded=0L; $bad=0
foreach($r in $rows){
 if($r.original_split -ne 'NOT_PROVIDED_IN_PUBLIC_INDEX' -or $r.dataset -ne 'NCSLGR'){throw 'ONLY_PREDECLARED_CUSTOM_NCSLGR_SPLIT_AUTHORIZED'}
 $uri=[Uri]$r.source_url
 if($uri.Scheme -ne 'https' -or $uri.DnsSafeHost -ne 'csr.bu.edu' -or $uri.UserInfo){throw 'NONPUBLIC_SOURCE_REJECTED'}
 $entry=[PSCustomObject]@{id=$r.sample_id;url=$r.source_url;kind='video';split=$r.split;target=$r.video_path;
   max_bytes=[long]$plan.download_max_bytes_per_file;expected_bytes=$null;expected_sha256=$null;
   max_expanded_bytes=0;annotation_type=$r.annotation_type;allowed_hosts=@('csr.bu.edu','www.bu.edu')}
 if($DryRun){continue}
 if(([DateTime]::UtcNow-$started).TotalSeconds -gt 600){throw 'ACQUISITION_WALL_BUDGET'}
 try{
   $result=Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 2 -TimeoutSeconds 60
   if($result.status -eq 'DOWNLOADED'){$downloaded += $result.bytes}
   if($downloaded -gt [long]$plan.download_max_total_bytes){throw 'ACTUAL_DOWNLOAD_TOTAL_CAP'}
   $result | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
   $bad++
   [PSCustomObject]@{id=$r.sample_id;status='FAILED';error=$_.Exception.Message;url=$r.source_url} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
[PSCustomObject]@{partition=$Partition;files=$rows.Count;max_planned_bytes=$plannedBytes;actual_downloaded_bytes=$downloaded;failures=$bad;dry_run=[bool]$DryRun;ledger=$ledger} | ConvertTo-Json
if($bad){throw "ACQUISITION_HAS_${bad}_FAILURES_NO_SILENT_CORPUS_REDUCTION"}
