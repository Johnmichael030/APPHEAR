$ErrorActionPreference='Stop'
$run=$PSScriptRoot;$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
& $python -B (Join-Path $run 'tools/patch_smoke_monitor.py')
if($LASTEXITCODE -ne 0){throw 'SMOKE_MONITOR_FIX_FAILED'}
if(Test-Path -LiteralPath (Join-Path $run 'pilot_outcome.json')){return}
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $run "logs/gpu_v2_$stamp.csv") -RedirectStandardError (Join-Path $run "logs/gpu_v2_$stamp.stderr.log") -WindowStyle Hidden -PassThru
$walls=@()
try{
 foreach($name in @('smoke','reference48')){
  $folder=if($name -eq 'smoke'){'smoke_v2'}else{$name}
  $arguments=@('-B',(Join-Path $run 'tools/train_pilot.v2.py'),'--name',$name)
  if(!(Test-Path -LiteralPath (Join-Path $run "models/$folder/training_status.json")) -and (Test-Path -LiteralPath (Join-Path $run "models/$folder/checkpoints/latest.json"))){$arguments+='--resume'}
  $watch=[Diagnostics.Stopwatch]::StartNew()
  & $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $run "logs/${name}_v2_$stamp.log")
  $code=$LASTEXITCODE;$watch.Stop();$walls+= [PSCustomObject]@{name=$name;seconds=$watch.Elapsed.TotalSeconds;exit_code=$code;arguments=$arguments}
  if($code -ne 0){throw "PILOT_V2_TRAINING_FAILED_$name"}
 }
 & $python -B (Join-Path $run 'tools/adjudicate_pilot.py')
 if($LASTEXITCODE -ne 0){throw 'PILOT_V2_ADJUDICATION_FAILED'}
}finally{
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 $walls | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run "logs/wall_v2_$stamp.json")
}
