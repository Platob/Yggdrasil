<#
.SYNOPSIS
    Build and install the Yggdrasil connector for Power BI Desktop and/or Excel.

.DESCRIPTION
    Builds the .mez file from a local checkout or by downloading the latest
    sources from GitHub, then copies it to the standard custom connector
    directories.

.PARAMETER Source
    Where to obtain the connector sources.
      Local  - use the powerquery/ folder next to this script (default).
      GitHub - download the latest main branch from
               https://github.com/Platob/Yggdrasil.

.PARAMETER Target
    Which application to install for.
      PowerBI - Power BI Desktop custom connectors folder.
      Excel   - Excel / Power Query custom connectors folder.
      Both    - both of the above (default).

.PARAMETER Branch
    GitHub branch to download when -Source GitHub. Default: main.

.PARAMETER Repo
    GitHub repository in "owner/repo" format. Default: Platob/Yggdrasil.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Source Local  -Target PowerBI
    .\install.ps1 -Source GitHub -Target Excel
    .\install.ps1 -Source GitHub -Branch main -Repo Platob/Yggdrasil
#>
[CmdletBinding()]
param(
    [ValidateSet('Local', 'GitHub')]
    [string]$Source = 'Local',

    [ValidateSet('PowerBI', 'Excel', 'Both')]
    [string]$Target = 'Both',

    [string]$Branch = 'main',

    [string]$Repo = 'Platob/Yggdrasil'
)

$ErrorActionPreference = 'Stop'

$connectorName = 'Yggdrasil'
$mezName       = "$connectorName.mez"

# -- Required connector files (relative to powerquery folder) --
$connectorFiles = @(
    "$connectorName.pq"
    "$connectorName.query.pq"
    "resources.resx"
    "${connectorName}16.png"
    "${connectorName}20.png"
    "${connectorName}24.png"
    "${connectorName}32.png"
)

# -- Resolve source directory --

function Resolve-LocalSource {
    <# Use the powerquery folder next to this script. #>
    $dir = $PSScriptRoot
    Write-Host "[Local] Using source: $dir" -ForegroundColor Cyan
    return $dir
}

function Resolve-GitHubSource {
    <# Download the repo archive and extract the powerquery folder. #>
    $archiveUrl = "https://github.com/$Repo/archive/refs/heads/$Branch.zip"
    $tempZip    = Join-Path $env:TEMP "yggdrasil-$Branch.zip"
    $extractDir = Join-Path $env:TEMP "yggdrasil-$Branch-extract"

    Write-Host "[GitHub] Downloading $archiveUrl ..." -ForegroundColor Cyan

    # Clean previous download
    if (Test-Path $tempZip)    { Remove-Item $tempZip    -Force }
    if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $archiveUrl -OutFile $tempZip -UseBasicParsing
    }
    catch {
        Write-Error ("Failed to download from GitHub.`n" +
                     "  URL:   $archiveUrl`n" +
                     "  Error: $_")
        exit 1
    }

    Write-Host "[GitHub] Extracting archive..." -ForegroundColor Cyan
    Expand-Archive -Path $tempZip -DestinationPath $extractDir -Force

    # GitHub archives contain a top-level folder like "Yggdrasil-main/"
    $topLevel = Get-ChildItem $extractDir -Directory | Select-Object -First 1
    if (-not $topLevel) {
        Write-Error "Archive extraction failed - no top-level folder found."
        exit 1
    }

    $pqDir = Join-Path $topLevel.FullName 'powerquery'
    if (-not (Test-Path $pqDir)) {
        Write-Error "powerquery/ folder not found in the downloaded archive."
        exit 1
    }

    Write-Host "[GitHub] Source: $pqDir" -ForegroundColor Cyan
    return $pqDir
}

# -- Build .mez from source directory --

function Build-Connector {
    param([string]$SourceDir)

    # Validate all required files exist
    foreach ($f in $connectorFiles) {
        $fullPath = Join-Path $SourceDir $f
        if (-not (Test-Path $fullPath)) {
            Write-Error "Missing required file: $fullPath"
            exit 1
        }
    }

    $tempZip = Join-Path $env:TEMP "$connectorName-build.zip"
    $mezPath = Join-Path $env:TEMP $mezName

    if (Test-Path $tempZip) { Remove-Item $tempZip -Force }
    if (Test-Path $mezPath) { Remove-Item $mezPath -Force }

    $absolutePaths = $connectorFiles | ForEach-Object { Join-Path $SourceDir $_ }

    Write-Host "Packaging $connectorName connector..." -ForegroundColor Cyan
    Compress-Archive -Path $absolutePaths -DestinationPath $tempZip -CompressionLevel Optimal
    Move-Item $tempZip $mezPath -Force

    $size = (Get-Item $mezPath).Length
    Write-Host "Built: $mezPath `($size bytes`)" -ForegroundColor Green

    return $mezPath
}

# -- Install to target directories --

function Install-Connector {
    param([string]$MezPath)

    $destinations = @()

    if ($Target -in @('PowerBI', 'Both')) {
        $destinations += Join-Path $env:USERPROFILE 'Documents\Power BI Desktop\Custom Connectors'
    }
    if ($Target -in @('Excel', 'Both')) {
        $destinations += Join-Path $env:USERPROFILE 'Documents\Microsoft Power Query\Custom Connectors'
    }

    foreach ($dest in $destinations) {
        if (-not (Test-Path $dest)) {
            New-Item -ItemType Directory -Path $dest -Force | Out-Null
            Write-Host "Created: $dest" -ForegroundColor DarkGray
        }
        Copy-Item $MezPath (Join-Path $dest $mezName) -Force
        Write-Host "Installed: $(Join-Path $dest $mezName)" -ForegroundColor Green
    }
}

# -- Main --

Write-Host ""
Write-Host "=======================================================" -ForegroundColor DarkCyan
Write-Host "  Yggdrasil Power Query Connector Installer"             -ForegroundColor White
Write-Host "  Source: $Source  |  Target: $Target"                    -ForegroundColor DarkGray
if ($Source -eq 'GitHub') {
    Write-Host "  Repo:   $Repo  |  Branch: $Branch"                 -ForegroundColor DarkGray
}
Write-Host "=======================================================" -ForegroundColor DarkCyan
Write-Host ""

# 1. Resolve sources
$sourceDir = if ($Source -eq 'GitHub') {
    Resolve-GitHubSource
} else {
    Resolve-LocalSource
}

# 2. Build
$mezPath = Build-Connector -SourceDir $sourceDir

# 3. Install
Install-Connector -MezPath $mezPath

# 4. Cleanup temp .mez
if (Test-Path $mezPath) { Remove-Item $mezPath -Force }

Write-Host ""
Write-Host "Done. Restart Power BI Desktop / Excel to load the connector." -ForegroundColor Cyan
Write-Host "Ensure 'Allow any extension to load' is enabled in Options > Security > Data Extensions." -ForegroundColor Yellow
Write-Host ""

