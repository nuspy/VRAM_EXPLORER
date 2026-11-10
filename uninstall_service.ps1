<#
.SYNOPSIS
    Rimuove il servizio VRAM Explorer.

.DESCRIPTION
    Ferma e disinstalla il servizio. I dati dell'utente (storico, configurazione,
    profili) restano in %LOCALAPPDATA%\VramExplorer e vanno rimossi
    esplicitamente con -RemoveData: cancellare mesi di storico come effetto
    collaterale di una disinstallazione sarebbe una brutta sorpresa.
#>

[CmdletBinding()]
param(
    [string]$ServiceName = "VramExplorer",
    [switch]$RemoveData
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Questo script va eseguito da un PowerShell con privilegi di amministratore."
}

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Host "Il servizio $ServiceName non risulta installato." -ForegroundColor Yellow
} else {
    Write-Host "==> Arresto del servizio" -ForegroundColor Cyan
    if ($svc.Status -ne "Stopped") {
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 3
    }

    $nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
    if ($nssm) {
        & $nssm remove $ServiceName confirm | Out-Null
    } else {
        & sc.exe delete $ServiceName | Out-Null
    }
    Write-Host "    servizio rimosso" -ForegroundColor Green
}

# Il processo che detiene la riserva di VRAM e' figlio del servizio: se e'
# sopravvissuto continuerebbe a occupare memoria senza che nulla lo governi.
$balloons = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "balloon_main\.py" }
foreach ($p in $balloons) {
    Write-Host "==> Rilascio della riserva di VRAM (pid $($p.ProcessId))" -ForegroundColor Cyan
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

if ($RemoveData) {
    $dataDir = Join-Path $env:LOCALAPPDATA "VramExplorer"
    if (Test-Path $dataDir) {
        Write-Host "==> Rimozione dei dati in $dataDir" -ForegroundColor Cyan
        Remove-Item -Recurse -Force $dataDir
        Write-Host "    dati rimossi" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "Storico e configurazione sono stati conservati in:" -ForegroundColor Yellow
    Write-Host "  $env:LOCALAPPDATA\VramExplorer"
    Write-Host "Per eliminarli:  .\uninstall_service.ps1 -RemoveData"
}
Write-Host ""
