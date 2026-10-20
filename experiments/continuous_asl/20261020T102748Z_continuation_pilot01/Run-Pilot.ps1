$ErrorActionPreference='Stop'
$repo=(Get-Item $PSScriptRoot).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
if(Test-Path -LiteralPath (Join-Path $PSScriptRoot 'training_status.json')){
 & $python -B (Join-Path $PSScriptRoot 'train_pilot.py')
 if($LASTEXITCODE -ne 0){throw 'COMPLETED_PILOT_VERIFICATION_FAILED'}
 return
}
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $PSScriptRoot "gpu_$stamp.csv") -RedirectStandardError (Join-Path $PSScriptRoot "gpu_$stamp.stderr.txt") -WindowStyle Hidden -PassThru
$watch=[Diagnostics.Stopwatch]::StartNew()
try{
 & $python -B (Join-Path $PSScriptRoot 'train_pilot.py') 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot "training_$stamp.log")
 $code=$LASTEXITCODE
 if($code -ne 0){throw "PILOT_FAILED_$code"}
}finally{
 $watch.Stop()
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 [PSCustomObject]@{seconds=$watch.Elapsed.TotalSeconds;exit_code=$code;command="$python -B $PSScriptRoot/train_pilot.py"} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot "command_wall_$stamp.json") -Encoding utf8
}
