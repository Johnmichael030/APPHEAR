$ErrorActionPreference='Stop'
$run=$PSScriptRoot
New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force | Out-Null
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$limits=Get-Content (Join-Path $run 'acquisition_limits.v2.json') -Raw | ConvertFrom-Json
$plan=Get-Content (Join-Path $run 'depaul_catalog/plan.json') -Raw | ConvertFrom-Json
$rows=@($plan.rows | Where-Object {!$_.existing_verified_utterance -and $_.role -eq 'NEW_RESEARCH_TRAIN_CANDIDATE'})
$groups=@($rows | Group-Object source_url | Sort-Object Count -Descending)
$ledger=Join-Path $run ('logs/depaul_media_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
foreach($group in $groups){
 # Semicolon-separated source entries can be temporal parts or alternate recordings.
 # Preserve them for a separately verified adapter, never guess which interpretation applies.
 if($group.Name.Contains(';')){
  [PSCustomObject]@{source_url=$group.Name;status='DEFERRED_SOURCE_PART_OR_ALTERNATIVE_SEMANTICS';utterances=$group.Count} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
  continue
 }
 $url=[Uri]$group.Name
 if($url.Scheme -ne 'https' -or $url.Host -ne 'csr.bu.edu' -or $url.UserInfo){throw 'UNVERIFIED_SOURCE_URL'}
 $key=(Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($url.AbsoluteUri))) -Algorithm SHA256).Hash.ToLowerInvariant().Substring(0,20)
 $id='depaul_media_'+$key
 $target=Join-Path $run "data/raw/ncslgr_media/$key.mov"
 if((Test-Path -LiteralPath (Join-Path $run "logs/download_${id}_attempt1.json")) -and !(Test-Path -LiteralPath $target)){continue}
 $oldRun=Join-Path $repo 'experiments/continuous_asl/20261020_continuous_expand04'
 $newBytes=[long]((Get-ChildItem (Join-Path $oldRun 'data/raw') -File -Recurse | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $here=[long]((Get-ChildItem (Join-Path $run 'data/raw') -File -Recurse | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $left=[long]$limits.total_acquired_payload_cap_bytes-1895291702L-$newBytes-$here
 if($left -le 0){throw 'AUTHORIZED_TOTAL_ACQUISITION_CAP'}
 if((Get-PSDrive D).Free -lt [long]$limits.minimum_disk_headroom_bytes){throw 'DISK_HEADROOM'}
 $entry=[PSCustomObject]@{id=$id;url=$url.AbsoluteUri;kind='video';split='RESEARCH_TRAIN_ONLY_COLLECTIONS';target=$target;
  max_bytes=[Math]::Min([long]$limits.single_recording_cap_bytes,$left);expected_bytes=$null;expected_sha256=$null;
  max_expanded_bytes=0;annotation_type='AUTHENTIC_ELAN_MAIN_GLOSS';allowed_hosts=@('csr.bu.edu','www.bu.edu')}
 try{
  $outcome=Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 2 -TimeoutSeconds $limits.single_file_timeout_seconds
  [PSCustomObject]@{id=$id;status=$outcome.status;source_url=$url.AbsoluteUri;target=$target;bytes=$outcome.bytes;sha256=$outcome.sha256;candidate_utterances=$group.Count} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
  [PSCustomObject]@{id=$id;status='FAILED';source_url=$url.AbsoluteUri;error=$_.Exception.Message;candidate_utterances=$group.Count} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
Get-Content -LiteralPath $ledger | ForEach-Object {$_ | ConvertFrom-Json} | Group-Object status | Select-Object Name,Count | ConvertTo-Json
