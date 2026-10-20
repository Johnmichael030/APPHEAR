param([ValidateSet('Status','AcquireTrainDev','PrepareTrainDev','Smoke','Reference','Candidate','Freeze','AcquireTest','PrepareTest','Evaluate','Tests','Verify')][string]$Phase='Status',[switch]$Resume)
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
if($Phase -eq 'Status'){
 foreach($name in @('smoke','reference48','compact24')){
  $status=Join-Path $run "models/$name/training_status.json"
  if(Test-Path $status){$s=Get-Content $status -Raw | ConvertFrom-Json;[PSCustomObject]@{model=$name;status=$s.status;epochs=$s.completed_epochs;best=$s.best_dev_score} | ConvertTo-Json -Compress}
 }
 if(Test-Path (Join-Path $run 'final_results.json')){Get-Content (Join-Path $run 'final_results.json') -Raw | ConvertFrom-Json | Select-Object rows,signer_independence | ConvertTo-Json}
 return
}
if($Phase -eq 'Verify'){
 $manifest=Get-Content (Join-Path $run 'artifact_manifest.json') -Raw | ConvertFrom-Json
 foreach($entry in $manifest.files){
  if((Get-FileHash (Join-Path $run $entry.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256){throw "ARTIFACT_HASH: $($entry.path)"}
 }
 [PSCustomObject]@{status='PASS';artifacts=$manifest.files.Count;production_enabled=$false} | ConvertTo-Json
 return
}
$log=Join-Path $run ('logs/'+$Phase+'_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.txt')
$tool=Join-Path $run 'tools/model_run.py';$argv=@()
switch($Phase){
 'AcquireTrainDev' {& (Join-Path $run 'acquire.ps1') -Partition train_dev;return}
 'AcquireTest' {& (Join-Path $run 'acquire.ps1') -Partition test;return}
 'PrepareTrainDev' {$tool=Join-Path $run 'tools/core.py';$argv=@('prepare','--partition','train_dev')}
 'PrepareTest' {$tool=Join-Path $run 'tools/core.py';$argv=@('prepare','--partition','test')}
 'Smoke' {$argv=@('train','--name','smoke')}
 'Reference' {$argv=@('train','--name','reference48')}
 'Candidate' {$argv=@('train','--name','compact24')}
 'Freeze' {$argv=@('freeze')}
 'Evaluate' {$argv=@('evaluate')}
 'Tests' {
  $junit=Join-Path $run ('logs/focused_tests_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.xml')
  $tool='-m';$argv=@('pytest',(Join-Path $run 'tools/test_expansion.py'),'-q','--tb=short','-p','no:cacheprovider',('--junitxml='+$junit))
 }
}
if($Resume){$argv += '--resume'}
# After selection freeze, report completed training without launching an optimizer.
if($Phase -in @('Smoke','Reference','Candidate') -and (Test-Path (Join-Path $run 'model_selection_freeze.json'))){
 $name=@{Smoke='smoke';Reference='reference48';Candidate='compact24'}[$Phase]
 $status=Get-Content (Join-Path $run "models/$name/training_status.json") -Raw | ConvertFrom-Json
 $ptr=Get-Content (Join-Path $run "models/$name/checkpoints/best.json") -Raw | ConvertFrom-Json
 if((Get-FileHash (Join-Path $run "models/$name/$($ptr.path)") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ptr.sha256){throw 'FROZEN_CHECKPOINT_CHANGED'}
 [PSCustomObject]@{status='FROZEN_COMPLETE_NO_TRAINING';model=$name;epochs=$status.completed_epochs} | ConvertTo-Json
 return
}
& $python -B $tool @argv *> $log
$code=$LASTEXITCODE
Get-Content $log -Tail 8
if($code -ne 0){throw "${Phase}_EXIT_${code}; see $log"}
