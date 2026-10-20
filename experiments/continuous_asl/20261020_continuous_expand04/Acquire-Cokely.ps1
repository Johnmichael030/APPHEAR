param([ValidateSet('Annotations','Video')][string]$Phase='Annotations')
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$entries=if($Phase -eq 'Annotations'){
 @(
  @{id='cpc_pledge_eaf';article=1005;filename=1;file='cokely/pledge.eaf'},
  @{id='cpc_preamble_eaf';article=1001;filename=1;file='cokely/preamble.eaf'},
  @{id='cpc_ferraro_eaf';article=1004;filename=1;file='cokely/ferraro.eaf'}
 )
}else{
 if(!(Test-Path -LiteralPath (Join-Path $run 'cokely_annotation_review.json'))){throw 'ANNOTATIONS_MUST_BE_VERIFIED_FIRST'}
 $review=Get-Content (Join-Path $run 'cokely_annotation_review.json') -Raw | ConvertFrom-Json
 if($review.status -ne 'PASS'){throw 'INVALID_AUTHENTIC_ANNOTATION_PIPELINE'}
 @(
  @{id='cpc_pledge_sd';article=1005;filename=0;file='cokely/pledge_sd.mp4'},
  @{id='cpc_ferraro_sd';article=1004;filename=0;file='cokely/ferraro_sd.mp4'}
 )
}
$ledger=Join-Path $run ('logs/cokely_'+$Phase+'_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
foreach($entry in $entries){
 $maximum=if($Phase -eq 'Annotations'){2097152L}else{200000000L}
 $previous=1895291702L
 $current=[long]((Get-ChildItem (Join-Path $run 'data/raw') -File -Recurse -ErrorAction SilentlyContinue | Where-Object {$_.Extension -in @('.eaf','.mp4','.part')} | Measure-Object Length -Sum).Sum)
 $maximum=[Math]::Min($maximum,2147483648L-$previous-$current)
 if($maximum -le 0){throw 'GLOBAL_2GIB_CAP'}
 if((Get-PSDrive D).Free -lt 10737418240L){throw 'DISK_HEADROOM'}
 $url="https://encompass.eku.edu/cgi/viewcontent.cgi?article=$($entry.article)&context=cokely_videos&filename=$($entry.filename)&type=additional"
 $request=[PSCustomObject]@{id=$entry.id;url=$url;kind=$(if($Phase -eq 'Annotations'){'xml'}else{'video'});
  split='RESEARCH_POOL_UNASSIGNED';target=(Join-Path $run ('data/raw/'+$entry.file));max_bytes=$maximum;
  expected_bytes=$null;expected_sha256=$null;max_expanded_bytes=0;annotation_type='AUTHENTIC_ELAN_ID_GLOSS_WITH_UTTERANCE_BOUNDARIES';allowed_hosts=@('encompass.eku.edu')}
 try{
  $outcome=Invoke-PublicDownload -Entry $request -RunRoot $run -MaximumAttempts 1 -TimeoutSeconds 120
  $outcome | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
  [PSCustomObject]@{id=$entry.id;status='FAILED';source_url=$url;error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
Get-Content -LiteralPath $ledger
