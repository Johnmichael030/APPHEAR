$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
Import-Module (Join-Path $repo 'tools/asl_research_v2/DownloadV2.psm1') -Force
$priorRun=Join-Path $repo 'experiments/continuous_asl/20261020_continuous_expand04'
$newBytes=[long]((Get-ChildItem (Join-Path $priorRun 'data/raw') -File -Recurse | Where-Object {$_.Name -notmatch '\.(receipt|http)\.json$'} | Measure-Object Length -Sum).Sum)
$remaining=2147483648L-1895291702L-$newBytes
$entry=[PSCustomObject]@{id='depaul_elan_annotations';url='https://asl.cs.depaul.edu/corpus/elanBUcorpus.zip';kind='zip';
 split='ANNOTATION_ONLY_OFFICIAL_UNIVERSITY_CONVERSION';target=(Join-Path $run 'data/raw/depaul/elanBUcorpus.zip');
 max_bytes=[Math]::Min(33554432L,$remaining);expected_bytes=$null;expected_sha256=$null;max_expanded_bytes=134217728L;
 annotation_type='NCSLGR_ELAN_CONVERSION_AUTHENTIC_GLOSS_TIMES';allowed_hosts=@('asl.cs.depaul.edu')}
Invoke-PublicDownload -Entry $entry -RunRoot $run -MaximumAttempts 1 -TimeoutSeconds 120 | ConvertTo-Json
