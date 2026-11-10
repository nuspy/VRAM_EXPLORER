"""Azioni dirette sui processi: trim, sospensione, ripresa, terminazione.

Sono le vie di ultima istanza, quando il processo non offre un modo civile di
liberare la memoria. Il servizio non le usa mai da solo: arrivano sempre da un
comando esplicito dell'utente, e quelle distruttive richiedono una conferma che
nomina il processo.

La lista dei processi protetti non e' un'opinione: terminare `dwm.exe` o
`csrss.exe` fa cadere la sessione o l'intero sistema. Vengono mostrati e
analizzati come tutti gli altri, ma le azioni distruttive restano disabilitate
con la motivazione in chiaro.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes

import psutil

from ..core.models import ProcInfo
from ..collect.proc_source import is_protected

log = logging.getLogger("vramexp.actions")

PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_TERMINATE = 0x0001


def is_protected_process(info: ProcInfo) -> tuple[bool, str]:
    """(protetto, motivo). Il motivo va mostrato all'utente, non solo registrato."""
    if is_protected(info):
        return True, (f"«{info.name}» e' un componente di Windows: terminarlo "
                      "farebbe cadere la sessione o il sistema")
    if info.pid <= 4:
        return True, "processo del kernel"
    return False, ""


def perform_process_action(pid: int, kind: str) -> tuple[bool, str]:
    """Esegue l'azione richiesta. Non verifica i permessi di alto livello:
    quello spetta a `Unloader.execute`, che conosce il contesto."""
    if kind == "trim":
        return trim_working_set(pid)
    if kind == "suspend":
        return suspend_process(pid)
    if kind == "resume":
        return resume_process(pid)
    if kind == "kill":
        return kill_process(pid)
    return False, f"azione «{kind}» sconosciuta"


def trim_working_set(pid: int) -> tuple[bool, str]:
    """Riduce il working set del processo.

    Agisce sulla memoria di sistema; sulla VRAM l'effetto e' indiretto e non
    garantito, perche' puo' indurre il gestore della memoria a sfrattare
    allocazioni video non recentemente usate. Va presentato come tentativo, non
    come soluzione.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_LIMITED_INFORMATION,
                                  False, pid)
    if not handle:
        return False, f"impossibile aprire il processo {pid}: {ctypes.get_last_error()}"
    try:
        if not psapi.EmptyWorkingSet(wintypes.HANDLE(handle)):
            return False, f"EmptyWorkingSet fallita: {ctypes.get_last_error()}"
        return True, ("working set ridotto (l'effetto sulla VRAM e' indiretto "
                      "e puo' essere nullo)")
    finally:
        kernel32.CloseHandle(handle)


def suspend_process(pid: int) -> tuple[bool, str]:
    """Congela il processo tramite `NtSuspendProcess`.

    Un processo sospeso non usa la GPU, quindi l'OS puo' sfrattarne le
    allocazioni video quando serve memoria ad altri. E' reversibile, il che lo
    rende preferibile alla terminazione quando si vuole solo fare spazio
    temporaneamente.
    """
    ntdll = ctypes.WinDLL("ntdll")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        return False, f"impossibile aprire il processo {pid}: {ctypes.get_last_error()}"
    try:
        status = ntdll.NtSuspendProcess(wintypes.HANDLE(handle))
        if status != 0:
            return False, f"NtSuspendProcess ha restituito 0x{status & 0xFFFFFFFF:08X}"
        return True, ("processo sospeso: l'applicazione non risponde piu' finche' "
                      "non viene ripresa")
    finally:
        kernel32.CloseHandle(handle)


def resume_process(pid: int) -> tuple[bool, str]:
    ntdll = ctypes.WinDLL("ntdll")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        return False, f"impossibile aprire il processo {pid}: {ctypes.get_last_error()}"
    try:
        status = ntdll.NtResumeProcess(wintypes.HANDLE(handle))
        if status != 0:
            return False, f"NtResumeProcess ha restituito 0x{status & 0xFFFFFFFF:08X}"
        return True, "processo ripreso"
    finally:
        kernel32.CloseHandle(handle)


def kill_process(pid: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Termina il processo, provando prima la via garbata.

    `terminate()` su Windows equivale gia' a `TerminateProcess`, ma passare da
    psutil consente di attendere l'uscita effettiva e di riportare se il processo
    e' davvero morto invece di dichiararlo per fede.
    """
    try:
        p = psutil.Process(pid)
        name = p.name()
        p.terminate()
        try:
            p.wait(timeout=timeout)
            return True, f"«{name}» terminato"
        except psutil.TimeoutExpired:
            p.kill()
            p.wait(timeout=timeout)
            return True, f"«{name}» terminato forzatamente"
    except psutil.NoSuchProcess:
        return True, "il processo era gia' terminato"
    except psutil.AccessDenied:
        return False, ("accesso negato: il processo appartiene a un altro utente "
                       "o e' protetto. Serve l'esecuzione come amministratore.")
    except psutil.TimeoutExpired:
        return False, "il processo non e' terminato entro il tempo previsto"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def is_suspended(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_STOPPED
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
