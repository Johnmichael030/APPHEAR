#Requires -Version 7.0
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
Import-Module (Join-Path $PSScriptRoot 'Common.psm1') -Force
$run=Get-ExperimentRoot $RunRoot
$snapshots=[ordered]@{}
$queries=[ordered]@{
 'Get-CimInstance Win32_OperatingSystem'={Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory};
 'Get-CimInstance Win32_Processor'={Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors};
 'Get-CimInstance Win32_ComputerSystem'={Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,AutomaticManagedPagefile};
 'Get-CimInstance Win32_PageFileUsage'={Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage};
 'Get-CimInstance Win32_VideoController'={Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,Status};
 'Get-ComputerInfo selected OS properties'={Get-ComputerInfo -Property WindowsProductName,WindowsVersion,OsBuildNumber};
 'Get-Volume C,D'={Get-Volume -DriveLetter C,D | Select-Object DriveLetter,FileSystem,Size,SizeRemaining,HealthStatus};
 'Get-PSDrive filesystem'={Get-PSDrive -PSProvider FileSystem | Select-Object Name,Root,Used,Free};
 'Get-Command tool paths'={Get-Command nvidia-smi,python,python3,py,pip,uv,git,git-lfs,curl,wsl,ffmpeg,7z,nvcc,node,javac -ErrorAction SilentlyContinue | Select-Object Name,Source};
 'nvidia-smi exact resource query'={& nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,temperature.gpu --format=csv,noheader; if($LASTEXITCODE -ne 0){throw 'NVIDIA_SMI_FAILED'}};
 'git status'={& git -C (Join-Path $PSScriptRoot '../..') status --short 2>&1; if($LASTEXITCODE -ne 0){'No Git metadata in this copy; branch/commit UNAVAILABLE'}};
 'git and download versions'={& git --version; & git lfs version; & uv --version; & curl.exe --version};
 'isolated Python version'={& (Join-Path $run 'runtime/venv/Scripts/python.exe') --version; if($LASTEXITCODE -ne 0){throw 'PYTHON_VERSION_FAILED'}}
}
foreach($key in $queries.Keys){try{$snapshots[$key]=@{status='OK';output=((& $queries[$key] | Out-String -Width 300).Trim())}}catch{$snapshots[$key]=@{status='UNAVAILABLE';error=$_.Exception.Message}}}
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
Write-ExperimentJson @{utc=[DateTime]::UtcNow.ToString('o');powershell=$PSVersionTable.PSVersion.ToString();edition=$PSVersionTable.PSEdition;commands=$snapshots;framework_details='logs/gpu_probe.json and logs/framework_probe.json'} (Join-Path $run "logs/environment_$stamp.json") $run
Write-Output "Saved environment_$stamp.json; no installation or global configuration changes"
