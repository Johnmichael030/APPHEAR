param([ValidateSet('Pages','Exports','Videos')][string]$Phase='Pages')
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$source=Join-Path $repo 'experiments/continuous_asl/20261020_continuous_expand03'
$ledger=Join-Path $run ('logs/signstream_'+$Phase+'_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.jsonl')
$entries=@()
if($Phase -eq 'Pages'){
 # Public page candidates are named in the official corpus/index. Payload links are parsed only from successful pages.
 $catalog=Get-Content (Join-Path $source 'expanded_catalog_v2/plan.json') -Raw | ConvertFrom-Json
 $index=Get-Content (Join-Path $source 'dataset_plan.json') -Raw | ConvertFrom-Json
 $collections=Import-Csv -LiteralPath $index.index_path | Select-Object -ExpandProperty 'XML file' -Unique | Where-Object {$_ -match '^ncslgr10' -and $_ -notin @('ncslgr10e.xml','ncslgr10m.xml','ncslgr10p.xml','ncslgr10q.xml')}
 foreach($collection in $collections){
  $stem=[IO.Path]::GetFileNameWithoutExtension($collection)
  $entries+=@{id="bu_page_$stem";url="https://www.bu.edu/asllrp/cslgr/display/$stem/";kind='html';file="bu_signstream/pages/$stem.html";maximum=262144L}
 }
}elseif($Phase -eq 'Exports'){
 $targets=[Collections.Generic.HashSet[string]]::new()
 foreach($page in Get-ChildItem -LiteralPath (Join-Path $run 'data/raw/bu_signstream/pages') -Filter *.html){
  $base=[Uri]"https://www.bu.edu/asllrp/cslgr/display/$($page.BaseName)/"
  foreach($match in [regex]::Matches((Get-Content -LiteralPath $page.FullName -Raw),'(?is)href\s*=\s*["'']([^"'']*sstxt/[^"'']+)["'']')){
   $url=[Uri]::new($base,[System.Net.WebUtility]::HtmlDecode($match.Groups[1].Value))
   if($url.Scheme -eq 'http'){$url=[Uri]($url.AbsoluteUri -replace '^http:','https:')}
   if($url.Host -ne 'www.bu.edu' -or $url.AbsolutePath -notmatch '/sstxt/([0-9]+_[0-9]+)$'){throw 'UNEXPECTED_PUBLIC_EXPORT_LINK'}
   $key=$Matches[1]
   if($targets.Add($key)){$entries+=@{id="bu_export_$key";url=$url.AbsoluteUri;kind='text';file="bu_signstream/exports/$key.txt";maximum=65536L}}
  }
 }
 $plan=Join-Path $run 'signstream_export_links.json'
 if(!(Test-Path -LiteralPath $plan)){$entries | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $plan -Encoding utf8}
}else{
 $plan=Get-Content (Join-Path $run 'signstream_candidate_plan.json') -Raw | ConvertFrom-Json
 foreach($row in $plan.eligible_new){$entries+=@{id=$row.sample_id;url=$row.source_url;kind='video';file="ncslgr_structured/videos/$($row.sample_id).mov";maximum=8388608L}}
}
$start=[DateTime]::UtcNow
foreach($item in $entries){
 if(([DateTime]::UtcNow-$start).TotalSeconds -gt 900){throw 'BOUNDED_ACQUISITION_WALL_LIMIT_RESUME_REMAINING'}
 $target=Join-Path $run ('data/raw/'+$item.file)
 $failed=Join-Path $run ("logs/download_"+$item.id+"_attempt1.json")
 if((Test-Path -LiteralPath $failed) -and !(Test-Path -LiteralPath $target)){continue}
 $current=[long]((Get-ChildItem (Join-Path $run 'data/raw') -File -Recurse -ErrorAction SilentlyContinue | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $limit=[Math]::Min([long]$item.maximum,2147483648L-1895291702L-$current)
 if($limit -le 0){throw 'GLOBAL_2GIB_CAP'}
 if((Get-PSDrive D).Free -lt 10737418240L){throw 'DISK_HEADROOM'}
 $entry=[PSCustomObject]@{id=$item.id;url=$item.url;kind=$item.kind;split='UNASSIGNED_RESEARCH_POOL';target=$target;max_bytes=$limit;
  expected_bytes=$null;expected_sha256=$null;max_expanded_bytes=0;annotation_type='OFFICIAL_SIGNSTREAM_EXPORT_AND_LINKED_VIDEO';allowed_hosts=@('www.bu.edu','csr.bu.edu')}
 try{
  $result=Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 1 -TimeoutSeconds 45
  $result | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }catch{
  [PSCustomObject]@{id=$item.id;status='FAILED';source_url=$item.url;error=$_.Exception.Message} | ConvertTo-Json -Compress | Add-Content -LiteralPath $ledger -Encoding utf8
 }
}
$records=if(Test-Path -LiteralPath $ledger){@(Get-Content -LiteralPath $ledger | ForEach-Object {$_ | ConvertFrom-Json})}else{@()}
[PSCustomObject]@{phase=$Phase;planned=$entries.Count;outcomes=$records.Count;status_counts=@($records | Group-Object status | Select-Object Name,Count)} | ConvertTo-Json -Depth 4
