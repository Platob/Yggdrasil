<#
.SYNOPSIS
    Build the Yggdrasil Power Query custom connector (.mez file).

.DESCRIPTION
    Packages all connector files into a .mez archive that can be loaded
    by Power BI Desktop or Excel.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -OutputDir "C:\Users\me\Documents\Power BI Desktop\Custom Connectors"
#>
[CmdletBinding()]
param(
    [string]$OutputDir = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'

$connectorName = 'Yggdrasil'
$sourceDir     = $PSScriptRoot
$mezFile       = Join-Path $OutputDir "$connectorName.mez"
$tempZip       = Join-Path $env:TEMP "$connectorName.zip"

# Files to include in the .mez
$files = @(
    "$sourceDir\$connectorName.pq"
    "$sourceDir\$connectorName.query.pq"
    "$sourceDir\resources.resx"
    "$sourceDir\${connectorName}16.png"
    "$sourceDir\${connectorName}20.png"
    "$sourceDir\${connectorName}24.png"
    "$sourceDir\${connectorName}32.png"
)

# Validate all files exist
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        Write-Error "Missing file: $f"
        exit 1
    }
}

# Remove old artifacts
if (Test-Path $tempZip) { Remove-Item $tempZip -Force }
if (Test-Path $mezFile) { Remove-Item $mezFile -Force }

Write-Host "Packaging $connectorName connector..." -ForegroundColor Cyan

# Create zip
Compress-Archive -Path $files -DestinationPath $tempZip -CompressionLevel Optimal

# Rename to .mez
Move-Item $tempZip $mezFile -Force

$size = (Get-Item $mezFile).Length
Write-Host "Built: $mezFile `($size bytes`)" -ForegroundColor Green
Write-Host ""
Write-Host "To install:" -ForegroundColor Yellow
Write-Host "  Power BI: Copy to %USERPROFILE%\Documents\Power BI Desktop\Custom Connectors\"
Write-Host "  Excel:    Copy to %USERPROFILE%\Documents\Microsoft Power Query\Custom Connectors\"

