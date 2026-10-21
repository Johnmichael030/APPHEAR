Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ExperimentRoot {
    param([Parameter(Mandatory)][string]$RunRoot)
    $full = [IO.Path]::GetFullPath($RunRoot)
    $repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
    $parent = Join-Path $repo 'experiments/continuous_asl'
    if (!$full.StartsWith($parent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'RunRoot must be a named run under this repository experiments/continuous_asl'
    }
    if (!(Test-Path -LiteralPath (Join-Path $full 'preservation_baseline.json') -PathType Leaf)) {
        throw 'Preservation baseline is required before this phase'
    }
    return $full
}
function Assert-IsolatedOutput {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Root)
    $full = [IO.Path]::GetFullPath($Path)
    $base = [IO.Path]::GetFullPath($Root).TrimEnd('\','/')
    if (!$full.StartsWith($base + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output escapes isolated root: $full"
    }
    $ancestor = $full
    while ($ancestor -and $ancestor.Length -ge $base.Length) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Reparse points are not permitted in experiment outputs'
            }
        }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $full
}
function Write-ExperimentJson {
    param([Parameter(Mandatory)]$Value, [Parameter(Mandatory)][string]$Path,
          [Parameter(Mandatory)][string]$Root)
    $target = Assert-IsolatedOutput -Path $Path -Root $Root
    $null = [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $target -Encoding utf8NoBOM
}
function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
Export-ModuleMember -Function Get-ExperimentRoot,Assert-IsolatedOutput,Write-ExperimentJson,Get-FileSha256

