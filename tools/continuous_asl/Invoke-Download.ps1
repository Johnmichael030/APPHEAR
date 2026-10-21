#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,
      [string]$ManifestPath,[string[]]$Ids,[switch]$DryRun,[switch]$Extract,
      [long]$MaxTotalBytes=67108864,[string]$PythonPath)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
Import-Module (Join-Path $PSScriptRoot 'Download.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
if(!$ManifestPath){$ManifestPath=Join-Path $run 'acquisition_plan.json'}
$plan=@(Get-DownloadPlan $ManifestPath $run $Ids $MaxTotalBytes)
if($DryRun){
    $plan | Select-Object id,kind,split,max_bytes,max_expanded_bytes,target | ConvertTo-Json -Depth 4
    Write-Output ("Dry run only; maximum download bytes="+(($plan|Measure-Object max_bytes -Sum).Sum)+"; no network or filesystem mutations.")
    exit 0
}
$records=@()
try {
    foreach($entry in $plan){
        $result=Invoke-PublicDownload $entry $run
        $records+=[PSCustomObject]@{id=$entry.id;status=$result.status;source_url=$entry.url;
            path=$entry.target;bytes=$result.bytes;sha256=$result.sha256;official_hash_available=[bool]$entry.expected_sha256}
        if($Extract -and $entry.kind -in @('zip','gzip')){
            if(!$PythonPath){$PythonPath=Join-Path $run 'runtime/venv/Scripts/python.exe'}
            $destination=Assert-IsolatedOutput (Join-Path $run ('data/extracted/'+$entry.id)) (Join-Path $run 'data/extracted')
            if(Test-Path -LiteralPath $destination){throw 'Extraction target already exists; inspect retained results instead of overwriting'}
            $details=& $PythonPath -B (Join-Path $PSScriptRoot 'archive_safety.py') --archive $entry.target --max-expanded-bytes $entry.max_expanded_bytes --extract-to $destination --allowed-root (Join-Path $run 'data/extracted')
            if($LASTEXITCODE -ne 0){throw "Archive verification/extraction failed for $($entry.id)"}
            Write-ExperimentJson ($details|ConvertFrom-Json -AsHashtable) (Join-Path $run ('logs/extract_'+$entry.id+'.json')) $run
        }
        $result | ConvertTo-Json -Compress
    }
} finally {
    if($records.Count){
        $ledger=Assert-IsolatedOutput (Join-Path $run 'download_manifest.csv') $run
        $records | Export-Csv -LiteralPath $ledger -Append -NoTypeInformation -Encoding utf8NoBOM
    }
}

