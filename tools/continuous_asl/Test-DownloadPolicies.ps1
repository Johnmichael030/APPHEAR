#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
Import-Module (Join-Path $PSScriptRoot 'Download.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$fixture=Assert-IsolatedOutput (Join-Path $run 'regressions/download_policies_01') $run
if(Test-Path $fixture){throw 'FIXTURES_EXIST'}
$null=New-Item -ItemType Directory -Path $fixture
$results=[Collections.Generic.List[object]]::new()
function Expect-Rejection([string]$Name,[scriptblock]$Action){
    $rejected=$false
    try {& $Action}catch{$rejected=$true}
    if(!$rejected){throw "EXPECTED_REJECTION:$Name"}
    $results.Add(@{case=$Name;passed=$true})
}
$html=Join-Path $fixture 'error.zip';[IO.File]::WriteAllText($html,'<!doctype html><html>quota error</html>')
Expect-Rejection 'HTML_is_not_ZIP' {Assert-PayloadFormat $html zip}
Expect-Rejection 'Drive_quota_not_bypassed' {Get-PublicDriveConfirmation '<html>too many users quota</html>' 'https://drive.google.com/uc?id=abc'}
Expect-Rejection 'Drive_ID_change' {Get-PublicDriveConfirmation '<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="id" value="changed"></form>' 'https://drive.google.com/uc?id=original'}
$range=[Net.Http.Headers.ContentRangeHeaderValue]::new(64,99,100)
Assert-ResumeResponse 64 206 $range '"stable"' '"stable"' 100
$results.Add(@{case='valid_complete_resume';passed=$true})
Expect-Rejection 'ignored_range' {Assert-ResumeResponse 64 200 $range '"stable"' '"stable"' 100}
Expect-Rejection 'changed_validator' {Assert-ResumeResponse 64 206 $range '"old"' '"new"' 100}
Expect-Rejection 'missing_validator' {Assert-ResumeResponse 64 206 $range '"old"' '' 100}
Expect-Rejection 'short_range' {Assert-ResumeResponse 64 206 ([Net.Http.Headers.ContentRangeHeaderValue]::new(64,80,100)) '"same"' '"same"' 100}
Expect-Rejection 'range_over_budget' {Assert-ResumeResponse 64 206 $range '"same"' '"same"' 99}
Expect-Rejection 'unrequested_partial' {Assert-ResumeResponse 0 206 $range '' '' 100}
$base=@{id='fixture';url='https://example.invalid/data.zip';filename='fixture.zip';kind='zip';split='train';max_bytes=100;max_expanded_bytes=100;expected_bytes=$null;sha256=$null;annotation_type='regression_only';authorization='RESEARCH_METADATA_OR_INITIAL_SAMPLE'}
$manifest=Join-Path $fixture 'plan.json'
foreach($variant in @('test','http','host','escape','budget')){
    $item=$base.Clone()
    switch($variant){'test'{$item.split='test'}'http'{$item.url='http://example.invalid/data.zip'}'host'{$item.url='https://other.invalid/a'}'escape'{$item.filename='../../../../escape.zip'}'budget'{$item.max_bytes=1000}}
    @{schema='apphear.continuous_asl.download_plan.v1';allowed_hosts=@('example.invalid');items=@($item)} | ConvertTo-Json -Depth 5 | Set-Content $manifest -Encoding utf8NoBOM
    Expect-Rejection "manifest_$variant" {Get-DownloadPlan $manifest $run @() 100}
}
Write-ExperimentJson @{tests=$results.Count;passed=$true;results=$results;network_used=$false} (Join-Path $run 'logs/download_policy_tests.json') $run
Write-Output "$($results.Count) downloader policy tests passed"
