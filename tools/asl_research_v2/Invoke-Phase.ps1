#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[Parameter(Mandatory)][ValidateSet('Recover','TrainA','BaselineA','TrainB','How2Verify','Transfer','WarmStart','WarmRepeat','Compare','Reports','Verify')][string]$Phase,[switch]$Resume)
$ErrorActionPreference='Stop'
$repo=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$run=[IO.Path]::GetFullPath($RunRoot)
if(!$run.StartsWith((Join-Path $repo 'experiments/continuous_asl')+[IO.Path]::DirectorySeparatorChar)){throw 'RUN_SCOPE'}
$python=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
$scripts=@{Recover='recover_prepare.py';TrainA='train_a.py';BaselineA='baseline_a.py';TrainB='train_b.py';How2Verify='how2_verify.py';Transfer='transfer_legacy.py';WarmStart='warmstart_a.py';WarmRepeat='warmstart_repeat.py';Compare='compare.py';Reports='reports.py';Verify='verify.py'}
if($Resume -and $Phase -in @('TrainA','TrainB','WarmStart','WarmRepeat')){
    $folders=@{TrainA='experiment_A';TrainB='experiment_B';WarmStart='experiment_A/warmstart';WarmRepeat='experiment_A/warmstart_repeat'}
    $candidate=Join-Path $run $folders[$Phase]
    $statusFile=Join-Path $candidate 'training_status.json'
    if(Test-Path $statusFile){
        $state=Get-Content $statusFile -Raw|ConvertFrom-Json
        if($state.status -in @('MAX_EPOCHS','EARLY_STOP')){
            $pointer=Get-Content (Join-Path $candidate 'checkpoints/latest.json') -Raw|ConvertFrom-Json
            $checkpoint=[IO.Path]::GetFullPath((Join-Path $candidate $pointer.path))
            if(!$checkpoint.StartsWith($candidate+[IO.Path]::DirectorySeparatorChar) -or (Get-FileHash $checkpoint).Hash.ToLowerInvariant() -ne $pointer.sha256){throw 'COMPLETED_CHECKPOINT_IDENTITY'}
            Write-Output "$Phase VERIFIED_COMPLETE, $($state.completed_epochs) epochs; no training repeated."
            return
        }
    }
}
$arguments=@('-B',(Join-Path $PSScriptRoot $scripts[$Phase]),'--run-root',$run)
if($Resume){$arguments+='--resume'}
Push-Location -LiteralPath $repo
try{& $python @arguments;if($LASTEXITCODE -ne 0){throw "PHASE_FAILED:$Phase exit=$LASTEXITCODE"}}finally{Pop-Location}
