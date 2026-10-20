$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$plan=Get-Content (Join-Path $run 'full_index_v2/plan.json') -Raw | ConvertFrom-Json
$ledger=Join-Path $run ('logs/index_v2_acquisition_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
$start=[DateTime]::UtcNow
foreach($row in $plan.rows | Where-Object new_raw_required){
 if(([DateTime]::UtcNow-$start).TotalSeconds -gt 900){throw 'ACQUISITION_WALL_BOUND_RESUME_REMAINING'}
 $failed=Join-Path $run ("logs/download_indexv2_"+$row.sample_id+"_attempt1.json")
 if((Test-Path -LiteralPath $failed) -and !(Test-Path -LiteralPath $row.video_path)){continue}
 $current=[long]((Get-ChildItem (Join-Path $run 'data/raw') -File -Recurse | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $maximum=[Math]::Min(8388608L,2147483648L-1895291702L-$current)
 if($maximum -le 0){throw 'CONSERVATIVE_GLOBAL_2GIB_CAP'}
 if((Get-PSDrive D).Free -lt 10737418240L){throw 'DISK_HEADROOM'}
 $entry=[PSCustomObject]@{id=('indexv2_'+$row.sample_id);url=$row.source_url;kind='video';split=$row.split;
  target=$row.video_path;max_bytes=$maximum;expected_bytes=$null;expected_sha256=$null;max_expanded_bytes=0;
  annotation_type=$row.annotation_type;allowed_hosts=@('csr.bu.edu','www.bu.edu')}
 try{
  Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 1 -TimeoutSeconds 45 | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
  [PSCustomObject]@{id=$entry.id;status='FAILED';source_url=$entry.url;error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
$records=@(Get-Content -LiteralPath $ledger | ForEach-Object {$_ | ConvertFrom-Json})
$records | Group-Object status | Select-Object Name,Count | ConvertTo-Json
