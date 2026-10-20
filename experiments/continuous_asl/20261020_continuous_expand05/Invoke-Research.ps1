param([ValidateSet('Status','Tests','Smoke','Reference','Candidate')][string]$Phase='Status',[switch]$Resume)
$ErrorActionPreference='Stop'
$run=$PSScriptRoot
$repo=(Get-Item $run).Parent.Parent.Parent.FullName
$py=Join-Path $repo 'experiments/continuous_asl/20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
if($Phase -eq 'Status'){
 foreach($name in @('smoke','reference48','compact24')){
  $path=Join-Path $run "models/$name/training_status.json"
  if(Test-Path -LiteralPath $path){Get-Content -LiteralPath $path -Raw}
 }
 return
}
$null=New-Item -ItemType Directory -Path (Join-Path $run 'logs') -Force
$log=Join-Path $run ('logs/'+$Phase+'_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.txt')
if($Phase -eq 'Tests'){
 $xml=Join-Path $run ('logs/focused_'+(Get-Date -Format 'yyyyMMddTHHmmss_fff')+'.xml')
 & $py -B -m pytest (Join-Path $run 'tools/test_expanded.py') -q --tb=short -p no:cacheprovider "--junitxml=$xml" *> $log
}else{
 $name=@{Smoke='smoke';Reference='reference48';Candidate='compact24'}[$Phase]
 $arguments=@('--name',$name)
 if($Resume){$arguments+='--resume'}
 & $py -B (Join-Path $run 'tools/train_expanded.py') @arguments *> $log
}
$exitCode=$LASTEXITCODE
Get-Content -LiteralPath $log -Tail 5
if($exitCode -ne 0){throw "${Phase}_FAILED_$exitCode; $log"}
