<#
simforest - Hermis experiment launcher (PowerShell wrapper)

Usage:
  .\simforest.ps1 --newconfig.yaml
  .\simforest.ps1 configs\newconfig.yaml

This calls the Python launcher file "simforest" located next to this script.
#>

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $scriptDir 'simforest') @args
