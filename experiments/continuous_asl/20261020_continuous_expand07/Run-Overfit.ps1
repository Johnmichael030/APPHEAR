$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
$monitor=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList '--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader -lms 200' -RedirectStandardOutput (Join-Path $run "logs/overfit_gpu_$stamp.csv") -RedirectStandardError (Join-Path $run "logs/overfit_gpu_$stamp.stderr.log") -WindowStyle Hidden -PassThru
$arguments=@('-B',(Join-Path $run 'tools/overfit.py'))
if(!(Test-Path -LiteralPath (Join-Path $run 'overfit_diagnostic/result.json')) -and (Test-Path -LiteralPath (Join-Path $run 'overfit_diagnostic/models/latest.json'))){$arguments+='--resume'}
$watch=[Diagnostics.Stopwatch]::StartNew()
try{
 & $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $run "logs/overfit_$stamp.log")
 $code=$LASTEXITCODE
 if($code -ne 0){throw "OVERFIT_DIAGNOSTIC_FAILED_EXIT_$code"}
}finally{
 $watch.Stop()
 if(!$monitor.HasExited){Stop-Process -Id $monitor.Id}
 [PSCustomObject]@{seconds=$watch.Elapsed.TotalSeconds;command=$arguments;exit_code=$code} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run "logs/overfit_wall_$stamp.json")
}
