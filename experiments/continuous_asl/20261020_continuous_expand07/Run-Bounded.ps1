$ErrorActionPreference='Stop'
$run=$PSScriptRoot
if(Test-Path -LiteralPath (Join-Path $run 'diagnostic_protocol.v2.json')){throw 'WARMSTART_DRAFT_SUPERSEDED: use Run-Overfit.ps1; the main run requires a new canonical snapshot and experiment.'}
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force | Out-Null
& $python -B (Join-Path $run 'tools/initialize.py')
if($LASTEXITCODE -ne 0){throw 'INITIALIZATION_FAILED'}
if(Test-Path -LiteralPath (Join-Path $run 'comparison_dev.json')){
 & $python -B (Join-Path $run 'tools/train_warm.py') --name warm48
 if($LASTEXITCODE -ne 0){throw 'COMPLETED_CHECKPOINT_VERIFICATION_FAILED'}
 return
}
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
& $python -B -m pytest -q (Join-Path $run 'tools/test_warm_contract.py') "--junitxml=$run/logs/warm_tests_$stamp.xml"
if($LASTEXITCODE -ne 0){throw 'WARM_START_CONTRACT_TESTS_FAILED'}
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $run "logs/gpu_samples_$stamp.csv") -RedirectStandardError (Join-Path $run "logs/gpu_samples_$stamp.stderr.log") -WindowStyle Hidden -PassThru
$walls=@()
try{
 foreach($name in @('smoke','warm48')){
  $arguments=@('-B',(Join-Path $run 'tools/train_warm.py'),'--name',$name)
  if(!(Test-Path -LiteralPath (Join-Path $run "models/$name/training_status.json")) -and (Test-Path -LiteralPath (Join-Path $run "models/$name/checkpoints/latest.json"))){$arguments+='--resume'}
  $watch=[Diagnostics.Stopwatch]::StartNew()
  & $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $run "logs/${name}_$stamp.log")
  $code=$LASTEXITCODE;$watch.Stop()
  $walls+= [PSCustomObject]@{model=$name;seconds=$watch.Elapsed.TotalSeconds;exit_code=$code;arguments=$arguments}
  if($code -ne 0){throw "TRAINING_FAILED_$name"}
 }
 & $python -B (Join-Path $run 'tools/freeze_compare_warm.py')
 if($LASTEXITCODE -ne 0){throw 'COMPARISON_FAILED'}
}finally{
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 $walls | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run "logs/command_wall_$stamp.json") -Encoding utf8
}
