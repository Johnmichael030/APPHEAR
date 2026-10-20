$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
$six=Join-Path $repo 'experiments/continuous_asl/20261020_continuous_expand06'
New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
& $python -B (Join-Path $six 'tools/freeze_corpus.py') --watch 2>&1 | Tee-Object -FilePath (Join-Path $run "logs/corpus_freeze_$stamp.log")
if($LASTEXITCODE -ne 0){throw 'CANONICAL_CORPUS_FREEZE_FAILED'}
& $python -B (Join-Path $run 'tools/initialize.py')
if($LASTEXITCODE -ne 0){throw 'MAIN_DATA_INITIALIZATION_FAILED'}
if(Test-Path -LiteralPath (Join-Path $run 'comparison_dev.json')){
 & $python -B (Join-Path $run 'tools/train_main.py') --name reference48
 if($LASTEXITCODE -ne 0){throw 'COMPLETED_CHECKPOINT_HASH_FAILED'}
 return
}
& $python -B -m pytest -q (Join-Path $run 'tools/test_main_gates.py') "--junitxml=$run/logs/main_gates_$stamp.xml"
if($LASTEXITCODE -ne 0){throw 'MAIN_DATA_TESTS_FAILED'}
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $run "logs/gpu_samples_$stamp.csv") -RedirectStandardError (Join-Path $run "logs/gpu_samples_$stamp.stderr.log") -WindowStyle Hidden -PassThru
$walls=@()
try{
 foreach($name in @('smoke','reference48')){
  $arguments=@('-B',(Join-Path $run 'tools/train_main.py'),'--name',$name)
  if(!(Test-Path -LiteralPath (Join-Path $run "models/$name/training_status.json")) -and (Test-Path -LiteralPath (Join-Path $run "models/$name/checkpoints/latest.json"))){$arguments+='--resume'}
  $watch=[Diagnostics.Stopwatch]::StartNew()
  & $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $run "logs/${name}_$stamp.log")
  $code=$LASTEXITCODE;$watch.Stop()
  $walls+= [PSCustomObject]@{model=$name;seconds=$watch.Elapsed.TotalSeconds;exit_code=$code;arguments=$arguments}
  if($code -ne 0){throw "MAIN_TRAINING_FAILED_$name"}
 }
 & $python -B (Join-Path $run 'tools/compare_main.py')
 if($LASTEXITCODE -ne 0){throw 'POST_TRAIN_COMPARISON_FAILED'}
}finally{
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 $walls | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $run "logs/command_wall_$stamp.json") -Encoding utf8
}
