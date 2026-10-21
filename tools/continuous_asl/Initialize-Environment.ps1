#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot,[switch]$CorePackages,[switch]$PerceptionPackages,[switch]$ExportPackages)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$cache=Assert-IsolatedOutput (Join-Path $run 'cache/uv') $run
$pythonRoot=Assert-IsolatedOutput (Join-Path $run 'runtime/python') $run
$venv=Assert-IsolatedOutput (Join-Path $run 'runtime/venv') $run
$drive=[IO.DriveInfo]::new([IO.Path]::GetPathRoot($run))
if($drive.AvailableFreeSpace -lt 16GB){throw 'Reserve at least16GiB before installing the isolated ML stack'}
$uv=(Get-Command uv -ErrorAction Stop).Source
$python=Join-Path $venv 'Scripts/python.exe'
if(!(Test-Path -LiteralPath $python)){
    & $uv python install 3.11.11 --install-dir $pythonRoot --no-bin --no-registry --cache-dir $cache
    if($LASTEXITCODE -ne 0){throw 'Isolated Python installation failed'}
    $managed=Join-Path $pythonRoot 'cpython-3.11.11-windows-x86_64-none/python.exe'
    & $uv venv $venv --python $managed --no-project --no-python-downloads --cache-dir $cache
    if($LASTEXITCODE -ne 0){throw 'Isolated virtualenv creation failed'}
}
& $uv pip install --python $python 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128 --cache-dir $cache
if($LASTEXITCODE -ne 0){throw 'Official CUDA PyTorch installation failed'}
if($CorePackages -or $PerceptionPackages){
    & $uv pip install --python $python 'numpy==1.26.4' 'scipy==1.13.1' 'PyYAML==6.0.2' 'pytest==8.3.5' --cache-dir $cache
    if($LASTEXITCODE -ne 0){throw 'Core research package installation failed'}
}
if($PerceptionPackages){
    & $uv pip install --python $python 'opencv-python==4.11.0.86' 'onnxruntime==1.22.0' 'tensorflow==2.20.0' 'keras==3.15.0' --cache-dir $cache
    if($LASTEXITCODE -ne 0){throw 'Perception package installation failed'}
}
if($ExportPackages){
    & $uv pip install --python $python 'onnx==1.19.0' --cache-dir $cache
    if($LASTEXITCODE -ne 0){throw 'ONNX export package installation failed'}
}
$lock=Assert-IsolatedOutput (Join-Path $run 'requirements.observed.txt') $run
& $uv pip freeze --python $python | Set-Content -LiteralPath $lock -Encoding utf8NoBOM
if($LASTEXITCODE -ne 0){throw 'Environment freeze failed'}
Write-Output "Isolated interpreter: $python"

