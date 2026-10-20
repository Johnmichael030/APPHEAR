$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$py=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
& $py -B (Join-Path $run 'tools/initialize.py')
if($LASTEXITCODE -ne 0){throw "STAGING_FAILED_$LASTEXITCODE"}
& (Join-Path $run 'Invoke-Research.ps1') -Phase Tests
$stamp=Get-Date -Format 'yyyyMMddTHHmmss_fff'
$gpuLog=Join-Path $run "logs/gpu_samples_$stamp.csv"
$sampler=$null
$commands=@()
try{
 $sampler=Start-Process -FilePath 'nvidia-smi.exe' -ArgumentList @('--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits','--loop-ms=200',"--filename=$gpuLog") -WindowStyle Hidden -PassThru
 foreach($phase in @('Smoke','Reference','Candidate')){
  $watch=[Diagnostics.Stopwatch]::StartNew()
  & (Join-Path $run 'Invoke-Research.ps1') -Phase $phase
  $watch.Stop()
  $commands+=[PSCustomObject]@{command="& '$run/Invoke-Research.ps1' -Phase $phase";measured_command_wall_seconds=$watch.Elapsed.TotalSeconds;status='EXIT_ZERO'}
 }
}finally{
 if($sampler -and !$sampler.HasExited){Stop-Process -Id $sampler.Id -ErrorAction SilentlyContinue}
 [PSCustomObject]@{gpu_sampler_pid=$(if($sampler){$sampler.Id}else{$null});gpu_log=$gpuLog;interval_ms=200;commands=$commands} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $run "logs/command_wall_$stamp.json") -Encoding utf8
}
& $py -B (Join-Path $run 'tools/freeze_compare.py')
if($LASTEXITCODE -ne 0){throw "FROZEN_DEV_COMPARISON_FAILED_$LASTEXITCODE"}
