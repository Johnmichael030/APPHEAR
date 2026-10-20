param([switch]$UtteranceArchive)
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force | Out-Null
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$limits=Get-Content (Join-Path $run 'acquisition_limits.v2.json') -Raw | ConvertFrom-Json
$items=@([PSCustomObject]@{id='asllrp2025_csv';url='https://dai.cs.rutgers.edu/asllvd/signbank/asllrp_sentence_signs_2025_06_28.csv';file='asllrp_sentence_signs_2025_06_28.csv';kind='csv';maximum=8388608L;expanded=0L})
if($UtteranceArchive){
 if(!(Test-Path -LiteralPath (Join-Path $run 'asllrp2025_catalog/annotation_audit.json'))){throw 'VERIFY_ANNOTATIONS_BEFORE_VIDEO_ARCHIVE'}
 $audit=Get-Content (Join-Path $run 'asllrp2025_catalog/annotation_audit.json') -Raw | ConvertFrom-Json
 if($audit.status -ne 'PASS_ORDERED_DOMINANT_GLOSS_CANDIDATES'){throw 'ANNOTATION_GATE'}
 $items+= [PSCustomObject]@{id='asllrp2025_utterance_v3';url='https://dai.cs.rutgers.edu/asllvd/signbank/asllrp_videos/batch_utterance_video_v3_1.zip';file='batch_utterance_video_v3_1.zip';kind='zip';maximum=2147483648L;expanded=4294967296L}
}
foreach($item in $items){
 $four=Join-Path $repo 'experiments/continuous_asl/20261020_continuous_expand04/data/raw'
 $other=[long]((Get-ChildItem $four -Recurse -File | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $current=[long]((Get-ChildItem (Join-Path $run 'data/raw') -Recurse -File | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
 $left=[long]$limits.total_acquired_payload_cap_bytes-1895291702L-$other-$current
 if($left -lt $item.maximum){throw 'TOTAL_ACQUISITION_CAP'}
 if((Get-PSDrive D).Free -lt ($item.maximum+$item.expanded+[long]$limits.minimum_disk_headroom_bytes)){throw 'DISK_HEADROOM'}
 $entry=[PSCustomObject]@{id=$item.id;url=$item.url;kind=$item.kind;split='RESEARCH_CORPUS_NO_OFFICIAL_SPLITS_DEFINED';
  target=(Join-Path $run ('data/raw/asllrp2025/'+$item.file));max_bytes=$item.maximum;expected_bytes=$null;expected_sha256=$null;
  max_expanded_bytes=$item.expanded;annotation_type='AUTHENTIC_DOMINANT_HAND_ORDERED_OCCURRENCE_GLOSS_EVENTS';allowed_hosts=@('dai.cs.rutgers.edu')}
 Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 2 -TimeoutSeconds 900 | ConvertTo-Json
}
