param([switch]$DryRun)
$ErrorActionPreference='Stop'
$run=$PSScriptRoot;$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$plan=Get-Content (Join-Path $run 'expanded_catalog_v2/plan.json') -Raw | ConvertFrom-Json
if((Get-FileHash (Join-Path $run 'dataset_plan.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.frozen_v1_plan_sha256){throw 'FROZEN_TRAINING_PLAN_CHANGED'}
if((Get-FileHash (Join-Path $run 'model_selection_freeze.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $plan.frozen_v1_selection_sha256){throw 'MODEL_SELECTION_CHANGED'}
$new=@($plan.rows | Where-Object new_in_catalog_v2)
$rawBytes=[long]((Get-ChildItem (Join-Path $run 'data/raw/ncslgr') -Recurse -Filter *.mov | Measure-Object Length -Sum).Sum)
$used=[long]$plan.how2sign_bytes_reused_stat_only+[long]$plan.prior_ncslgr_raw_bytes+$rawBytes
if((Get-PSDrive D).Free -lt 10737418240L){throw 'DISK_HEADROOM'}
$ledger=Join-Path $run ('logs/catalog_acquisition_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
$bytes=0L;$bad=0;$started=[DateTime]::UtcNow
foreach($row in $new){
 if($DryRun){continue}
 if(([DateTime]::UtcNow-$started).TotalSeconds -gt 900){throw 'ACQUISITION_WALL_BUDGET'}
 $left=[Math]::Min([long]$plan.new_run_raw_total_cap-$rawBytes,[long]$plan.conservative_global_raw_cap_with_How2Sign-$used)
 if($left -le 0){throw 'ACQUISITION_TOTAL_CAP_NO_MORE_DOWNLOADS'}
 $uri=[Uri]$row.source_url
 if($uri.Scheme -ne 'https' -or $uri.DnsSafeHost -ne 'csr.bu.edu' -or $uri.UserInfo){throw 'ONLY_VERIFIED_OFFICIAL_INDEX_SOURCE_ALLOWED'}
 $entry=[PSCustomObject]@{id=$row.sample_id;url=$row.source_url;kind='video';split=$row.split;target=$row.video_path;
 max_bytes=[Math]::Min([long]$plan.per_file_download_cap,$left);expected_bytes=$null;expected_sha256=$null;
 max_expanded_bytes=0;annotation_type=$row.annotation_type;allowed_hosts=@('csr.bu.edu','www.bu.edu')}
 try{
  $result=Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 2 -TimeoutSeconds 60
  if($result.status -eq 'DOWNLOADED'){$bytes+=$result.bytes;$rawBytes+=$result.bytes;$used+=$result.bytes}
  $result | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
  $bad++;[PSCustomObject]@{id=$row.sample_id;status='FAILED';url=$row.source_url;error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
[PSCustomObject]@{requested=$new.Count;new_bytes=$bytes;failures=$bad;conservative_total_raw_bytes=$used;global_cap=$plan.conservative_global_raw_cap_with_How2Sign;dry_run=[bool]$DryRun} | ConvertTo-Json
if($bad){throw "CATALOG_HAS_${bad}_ACQUISITION_FAILURES; process other verified files and retain this ledger"}
