<#
.SYNOPSIS
    Installa VRAM Explorer come servizio Windows con avvio automatico e riavvio
    in caso di crash.

.DESCRIPTION
    Usa NSSM come supervisore. Il servizio gira come l'utente interattivo, non
    come SYSTEM: i contatori di prestazione per processo e le API locali dei
    backend (LM Studio, Ollama, ComfyUI) sono visibili correttamente solo nel
    contesto dell'utente che possiede quei processi.

    La resilienza e' su due livelli, di proposito ridondanti:
      * NSSM riavvia il processo quando termina in modo anomalo;
      * il Gestore servizi di Windows riavvia il servizio se NSSM stesso cade.

.PARAMETER Credential
    Credenziali dell'account sotto cui eseguire il servizio. Se omesse vengono
    chieste in modo interattivo. Con un account Microsoft occorre la password
    dell'account, non il PIN di Windows Hello.

.EXAMPLE
    .\install_service.ps1
    .\install_service.ps1 -ServiceName VramExplorer -Port 8765
#>

[CmdletBinding()]
param(
    [string]$ServiceName = "VramExplorer",
    [int]$Port = 8765,
    [System.Management.Automation.PSCredential]$Credential,
    [switch]$RunAsSystem,
    [string]$PythonPath
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "    $text" -ForegroundColor Green }
function Write-Warn2($text){ Write-Host "    $text" -ForegroundColor Yellow }

# --- verifiche preliminari -------------------------------------------------
Write-Step "Verifica dei prerequisiti"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Questo script va eseguito da un PowerShell con privilegi di amministratore."
}

$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) {
    throw @"
NSSM non trovato nel PATH.
Installarlo con uno dei seguenti comandi, poi riprovare:
    scoop install nssm
    choco install nssm
"@
}
Write-Ok "NSSM: $nssm"

if (-not $PythonPath) {
    $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $PythonPath -or -not (Test-Path $PythonPath)) {
    throw "Interprete Python non trovato. Indicarlo con -PythonPath."
}
Write-Ok "Python: $PythonPath"

$serviceScript = Join-Path $repo "service_main.py"
if (-not (Test-Path $serviceScript)) {
    throw "service_main.py non trovato in $repo"
}

# Verifica che l'ambiente sia realmente in grado di leggere la GPU: installare
# un servizio che non riesce a campionare sarebbe un fallimento silenzioso,
# scoperto solo consultando i log.
Write-Step "Verifica dell'accesso ai contatori GPU"
$check = & $PythonPath -c @"
import sys
sys.path.insert(0, r'$repo')
from vramexp.collect.pdh_source import PdhCollector
from vramexp.collect.nvml_source import NvmlSource
pdh = PdhCollector(); nv = NvmlSource()
print(f'PDH={pdh.available}|NVML={nv.available}|GPU={len(nv.gpu_info)}')
if pdh.init_error: print('PDH_ERR=' + pdh.init_error)
if nv.init_error: print('NVML_ERR=' + str(nv.init_error))
pdh.close(); nv.shutdown()
"@ 2>&1

Write-Host ($check | Out-String).Trim()
if ($check -notmatch "PDH=True") {
    Write-Warn2 "I contatori PDH non rispondono: la memoria per processo non sara' leggibile."
    Write-Warn2 "Provare a rigenerare i contatori con:  lodctr /R"
}

# --- credenziali -----------------------------------------------------------
if (-not $RunAsSystem) {
    if (-not $Credential) {
        Write-Step "Credenziali del servizio"
        Write-Host "    Il servizio gira come utente interattivo per vedere i processi"
        Write-Host "    e le API locali. Con un account Microsoft serve la password"
        Write-Host "    dell'account (non il PIN)."
        $default = "$env:COMPUTERNAME\$env:USERNAME"
        $Credential = Get-Credential -UserName $default `
            -Message "Account sotto cui eseguire $ServiceName"
    }
    if (-not $Credential) { throw "Credenziali non fornite: installazione annullata." }
}

# --- rimozione di un'installazione precedente ------------------------------
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Step "Rimozione dell'installazione precedente"
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    & $nssm remove $ServiceName confirm | Out-Null
    Start-Sleep -Seconds 2
    Write-Ok "servizio precedente rimosso"
}

# --- installazione ---------------------------------------------------------
Write-Step "Installazione del servizio $ServiceName"
& $nssm install $ServiceName $PythonPath "`"$serviceScript`" --port $Port" | Out-Null

& $nssm set $ServiceName DisplayName "VRAM Explorer" | Out-Null
& $nssm set $ServiceName Description `
    "Monitoraggio e gestione della memoria video. Osserva e segnala; non libera ne' termina processi di propria iniziativa." | Out-Null
& $nssm set $ServiceName AppDirectory $repo | Out-Null
& $nssm set $ServiceName Start SERVICE_AUTO_START | Out-Null

# Riavvio automatico. AppThrottle evita che un errore di configurazione produca
# un ciclo di riavvii a raffica: sotto i 10 secondi di vita NSSM rallenta.
& $nssm set $ServiceName AppExit Default Restart | Out-Null
& $nssm set $ServiceName AppRestartDelay 5000 | Out-Null
& $nssm set $ServiceName AppThrottle 10000 | Out-Null

# Log con rotazione: sotto servizio non c'e' una console da guardare.
$logDir = Join-Path $env:LOCALAPPDATA "VramExplorer\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
& $nssm set $ServiceName AppStdout (Join-Path $logDir "service-out.log") | Out-Null
& $nssm set $ServiceName AppStderr (Join-Path $logDir "service-err.log") | Out-Null
& $nssm set $ServiceName AppRotateFiles 1 | Out-Null
& $nssm set $ServiceName AppRotateOnline 1 | Out-Null
& $nssm set $ServiceName AppRotateBytes 8388608 | Out-Null

# Chiusura ordinata: il servizio chiude il database e rilascia la riserva.
& $nssm set $ServiceName AppStopMethodSkip 0 | Out-Null
& $nssm set $ServiceName AppStopMethodConsole 15000 | Out-Null

if (-not $RunAsSystem) {
    & $nssm set $ServiceName ObjectName $Credential.UserName `
        $Credential.GetNetworkCredential().Password | Out-Null
    Write-Ok "esecuzione come $($Credential.UserName)"
} else {
    Write-Warn2 "esecuzione come LocalSystem: i contatori per processo e le API"
    Write-Warn2 "dei backend potrebbero non essere visibili."
}

# Secondo livello di resilienza, indipendente da NSSM.
& sc.exe failure $ServiceName reset= 86400 `
    actions= restart/5000/restart/10000/restart/30000 | Out-Null
& sc.exe failureflag $ServiceName 1 | Out-Null
Write-Ok "riavvio automatico configurato (NSSM + Gestione servizi)"

# --- avvio -----------------------------------------------------------------
Write-Step "Avvio del servizio"
try {
    Start-Service -Name $ServiceName
} catch {
    Write-Warn2 "avvio fallito: $_"
    Write-Warn2 "controllare $logDir\service-err.log"
    throw
}

Start-Sleep -Seconds 6
$svc = Get-Service -Name $ServiceName
Write-Ok "stato: $($svc.Status)"

try {
    $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
    Write-Ok "API attiva — campioni raccolti: $($health.samples)"
} catch {
    Write-Warn2 "l'API non risponde ancora su 127.0.0.1:$Port"
    Write-Warn2 "verificare i log in $logDir"
}

Write-Host ""
Write-Host "Installazione completata." -ForegroundColor Green
Write-Host ""
Write-Host "  Interfaccia grafica :  pythonw `"$repo\vramexp_gui.py`""
Write-Host "  Riga di comando     :  python -m vramexp top"
Write-Host "  Log                 :  $logDir"
Write-Host "  Configurazione      :  $env:LOCALAPPDATA\VramExplorer\config.json"
Write-Host ""
Write-Host "  Disinstallazione    :  .\uninstall_service.ps1"
Write-Host ""
