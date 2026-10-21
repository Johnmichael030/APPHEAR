#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[switch]$Resume)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
if($Resume -and (Test-Path (Join-Path $run 'logs/training_status.json'))){
    $finished=Get-Content (Join-Path $run 'logs/training_status.json') -Raw | ConvertFrom-Json
    if($finished.status -in @('EARLY_STOP','MAX_EPOCHS')){
        Write-Output "Run already completed with $($finished.status) after $($finished.epochs_completed) epochs; no further training started."
        return
    }
}
if(!(Test-Path (Join-Path $run 'data/processed/full/quality.json'))){throw 'PREPROCESSING_REQUIRED'}
$python=Join-Path $run 'runtime/venv/Scripts/python.exe'
$script=Join-Path $PSScriptRoot 'ctc_experiment.py'
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$stdout=Join-Path $run "logs/train_$stamp.out.txt"
$stderr=Join-Path $run "logs/train_$stamp.err.txt"
$arguments='-B "'+$script+'" --run-root "'+$run+'"'
if($Resume){$arguments+=' --resume'}
$process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory (Split-Path (Split-Path $PSScriptRoot)) -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$monitor=Join-Path $run "logs/resources_$stamp.csv"
do {
    $process.Refresh()
    if(!$process.HasExited){
        $gpu=(& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null) -join ' '
        [PSCustomObject]@{utc=[DateTime]::UtcNow.ToString('o');pid=$process.Id;working_set_bytes=$process.WorkingSet64;peak_working_set_bytes=$process.PeakWorkingSet64;gpu=$gpu} | Export-Csv $monitor -Append -NoTypeInformation -Encoding utf8NoBOM
        Start-Sleep -Seconds 2
    }
} while(!$process.HasExited)
$process.WaitForExit()
@{exit_code=$process.ExitCode;stdout=$stdout;stderr=$stderr;resources=$monitor} | ConvertTo-Json | Set-Content (Join-Path $run "logs/train_exit_$stamp.json") -Encoding utf8NoBOM
Get-Content $stdout -Tail 3
if($process.ExitCode -ne 0){Get-Content $stderr -Tail 10;throw "TRAINING_FAILED:$($process.ExitCode); successful checkpoints retained"}
