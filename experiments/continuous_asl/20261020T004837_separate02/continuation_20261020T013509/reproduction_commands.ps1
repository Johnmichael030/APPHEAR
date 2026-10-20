param([ValidateSet('Verify','Matrix','CollectionPlan','ArchiveAudit')][string]$Phase='Verify')
$ErrorActionPreference='Stop'
$continuationRoot=$PSScriptRoot
switch($Phase){
 'Verify' {
   $manifest=Get-Content -LiteralPath (Join-Path $continuationRoot 'artifact_hashes.json') -Raw | ConvertFrom-Json
   $checked=0
   foreach($item in $manifest.files){
     $candidate=Join-Path $continuationRoot $item.path
     $actual=(Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
     if($actual -ne $item.sha256){throw "NEW_ARTIFACT_HASH_MISMATCH: $($item.path)"}
     $checked++
   }
   [PSCustomObject]@{status='PASS';new_artifacts_checked=$checked;training='NOT_STARTED';frozen_outputs='READ_ONLY'} | ConvertTo-Json
 }
 'Matrix' {Import-Csv -LiteralPath (Join-Path $continuationRoot 'data_readiness_matrix.csv') | Format-Table experiment,status,trainability,missing -Wrap}
 'CollectionPlan' {Get-Content -LiteralPath (Join-Path $continuationRoot 'collection_plan.md')}
 'ArchiveAudit' {
   # Optional diagnostic only if the archive/provenance may have changed; do not repeat routinely.
   $repo=(Get-Item -LiteralPath $continuationRoot).Parent.Parent.Parent.Parent.FullName
   $python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
   $output=Join-Path $continuationRoot ('recheck_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff'))
   & $python -B (Join-Path $continuationRoot 'audit.py') --out $output
   if($LASTEXITCODE -ne 0){throw "ARCHIVE_AUDIT_EXIT_$LASTEXITCODE"}
 }
}
