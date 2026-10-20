$ErrorActionPreference='Stop'
$repo=(Get-Item $PSScriptRoot).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
& $python -B (Join-Path $PSScriptRoot 'freeze_split.py') --watch
if($LASTEXITCODE -ne 0){throw 'NEW_SIGNER_SPLIT_GATE_FAILED'}
if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'pilot_fresh48/training_status.json')){
 & $python -B (Join-Path $PSScriptRoot 'train_fresh.py')
 if($LASTEXITCODE -ne 0){throw 'COMPLETED_FRESH_PILOT_HASH_FAILED'}
 return
}
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $PSScriptRoot "gpu_$stamp.csv") -RedirectStandardError (Join-Path $PSScriptRoot "gpu_$stamp.stderr.txt") -WindowStyle Hidden -PassThru
$watch=[Diagnostics.Stopwatch]::StartNew()
try{
 & $python -B (Join-Path $PSScriptRoot 'train_fresh.py') 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "training_$stamp.log")
 $code=$LASTEXITCODE
 if($code -ne 0){throw "FRESH_PILOT_FAILED_$code"}
}finally{
 $watch.Stop()
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 [PSCustomObject]@{seconds=$watch.Elapsed.TotalSeconds;exit_code=$code;command="$python -B $PSScriptRoot/train_fresh.py"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot "command_wall_$stamp.json") -Encoding utf8
}
