#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
Import-Module (Join-Path $PSScriptRoot 'Download.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$source=Join-Path $run 'data/raw/how2sign/download_how2sign.sh'
$meta=Get-Content ($source+'.part.http.json') -Raw | ConvertFrom-Json -AsHashtable
if(!$meta.validator){throw 'NO_REUSABLE_HTTP_VALIDATOR'}
$target=Assert-IsolatedOutput (Join-Path $run 'data/raw/regression_resume/download_how2sign.sh') (Join-Path $run 'data/raw')
if(Test-Path ($target+'.part')){throw 'PROBE_EXISTS'}
$null=[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
$prefix=[IO.File]::ReadAllBytes($source)[0..63]
[IO.File]::WriteAllBytes($target+'.part',$prefix)
Write-ExperimentJson $meta ($target+'.part.http.json') $run
$entry=[PSCustomObject]@{id='resume_probe';url=$meta.source_url;target=$target;kind='text';split='metadata';max_bytes=65536;expected_bytes=(Get-Item $source).Length;expected_sha256=$null;max_expanded_bytes=0;annotation_type='documentation_resume_regression';allowed_hosts=@('raw.githubusercontent.com')}
try {
    $result=Invoke-PublicDownload $entry $run -MaximumAttempts 1
    if((Get-FileSha256 $source) -ne $result.sha256){throw 'RESUMED_CONTENT_MISMATCH'}
    Write-ExperimentJson @{status='PASS';initial_partial_bytes=64;final_bytes=$result.bytes;sha256=$result.sha256;expected_hash_origin='previously_observed_payload_not_official';partial_source_retained=$true} (Join-Path $run 'logs/download_resume_test.json') $run
    $result | ConvertTo-Json -Compress
} catch {
    Write-ExperimentJson @{status='BLOCKED_OR_FAILED';reason=$_.Exception.Message;partial_retained=Test-Path ($target+'.part')} (Join-Path $run 'logs/download_resume_test.json') $run
    throw
}
