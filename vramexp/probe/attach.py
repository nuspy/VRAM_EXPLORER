"""Iniezione dell'agent in un processo Python gia' in esecuzione.

**Avvertenza, da leggere prima di usarlo.** Questa e' l'unica funzione del
progetto che scrive nella memoria di un altro processo e vi esegue codice. Non
esiste un modo per renderla sicura al 100%: se il processo target si trova in uno
stato particolare (GIL gia' in uso in modo anomalo, interprete in fase di
spegnimento, estensioni native che non rilasciano il GIL) l'iniezione puo'
bloccarlo o farlo terminare. Su un training di otto ore la perdita e' reale.

Per questo e': disattivata per default (`enable_hot_attach` in configurazione),
mai automatica, sempre preceduta da una conferma che nomina il processo, e
limitata ai processi dell'utente corrente che hanno gia' caricato una DLL di
CPython. La via raccomandata resta `VRAMEXP_AGENT=1` all'avvio, che non comporta
alcun rischio.

**Come funziona.** Non basta un `CreateRemoteThread` su `PyRun_SimpleString`,
perche' eseguire codice Python richiede di tenere il GIL. Serve quindi una
sequenza di tre chiamate — `PyGILState_Ensure`, `PyRun_SimpleString`,
`PyGILState_Release` — che viene realizzata con poche decine di byte di codice
macchina x64 iniettati insieme ai parametri.

Gli indirizzi delle funzioni nel processo remoto si calcolano per RVA: si carica
la stessa DLL di CPython in locale, si legge l'offset della funzione rispetto
alla base del modulo, e lo si somma alla base che quella DLL ha nel target.
L'ASLR sposta la base ma non gli offset interni.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import struct
import time
from ctypes import wintypes
from dataclasses import dataclass

import psutil

from . import protocol as P
from .client import AgentClient

log = logging.getLogger("vramexp.attach")

PROCESS_CREATE_THREAD = 0x0002
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_READ = 0x0010
INJECT_ACCESS = (PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION
                 | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ)

MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READ = 0x20
WAIT_OBJECT_0 = 0x0
WAIT_TIMEOUT = 0x102

# Sequenza x64: preserva rbx, riserva lo shadow space di 32 byte richiesto dalla
# convenzione Windows e mantiene rsp allineato a 16 a ogni call.
#   push rbx
#   sub  rsp, 0x20
#   mov  rbx, rcx                ; rcx = blocco parametri
#   call qword ptr [rbx]         ; PyGILState_Ensure()
#   mov  [rbx+0x18], rax         ; conserva lo stato del GIL
#   lea  rcx, [rbx+0x20]         ; indirizzo del codice sorgente
#   call qword ptr [rbx+0x08]    ; PyRun_SimpleString(code)
#   mov  rcx, [rbx+0x18]
#   call qword ptr [rbx+0x10]    ; PyGILState_Release(state)
#   add  rsp, 0x20
#   pop  rbx
#   xor  eax, eax
#   ret
SHELLCODE = bytes([
    0x53,
    0x48, 0x83, 0xEC, 0x20,
    0x48, 0x89, 0xCB,
    0xFF, 0x13,
    0x48, 0x89, 0x43, 0x18,
    0x48, 0x8D, 0x4B, 0x20,
    0xFF, 0x53, 0x08,
    0x48, 0x8B, 0x4B, 0x18,
    0xFF, 0x53, 0x10,
    0x48, 0x83, 0xC4, 0x20,
    0x5B,
    0x31, 0xC0,
    0xC3,
])
PARAM_HEADER_SIZE = 0x20      # tre puntatori + scratch per lo stato del GIL

_PY_DLL_RE = re.compile(r"^python(\d)(\d+)\.dll$", re.IGNORECASE)


@dataclass
class AttachResult:
    ok: bool
    message: str
    pid: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "message": self.message, "pid": self.pid,
                "detail": self.detail}


@dataclass
class _Target:
    pid: int
    python_dll_path: str
    remote_base: int


def check_eligibility(pid: int) -> tuple[bool, str]:
    """Verifica preliminare, prima di toccare qualunque cosa.

    Restituisce (idoneo, motivo). Il motivo va mostrato all'utente: sapere
    *perche'* un processo non e' iniettabile e' piu' utile di un rifiuto secco.
    """
    if pid == os.getpid():
        return False, "non ha senso iniettare l'agent nel processo che lo ospita"
    if pid <= 4:
        return False, "processo di sistema"

    try:
        proc = psutil.Process(pid)
        name = proc.name()
    except psutil.NoSuchProcess:
        return False, "il processo non esiste piu'"
    except psutil.AccessDenied:
        return False, "accesso negato: il processo appartiene a un altro utente"

    try:
        if proc.username() != psutil.Process(os.getpid()).username():
            return False, ("il processo appartiene a un altro utente: "
                           "l'iniezione richiederebbe privilegi elevati")
    except (psutil.AccessDenied, OSError):
        return False, "impossibile verificare il proprietario del processo"

    target = _find_python_dll(pid)
    if target is None:
        return False, (f"«{name}» non ha una DLL di CPython caricata: "
                       "non e' un processo Python iniettabile")

    if not _bitness_matches(target.python_dll_path):
        return False, "architettura incompatibile fra il target e questo interprete"

    if AgentClient().has_agent(pid):
        return False, "l'agent e' gia' attivo in questo processo"

    return True, f"processo Python idoneo ({os.path.basename(target.python_dll_path)})"


def attach(pid: int, confirmed: bool = False, enabled: bool = False,
           timeout_s: float = 10.0) -> AttachResult:
    """Inietta l'agent nel processo indicato.

    `enabled` riflette la configurazione (`enable_hot_attach`) e `confirmed` la
    conferma esplicita dell'utente: entrambe devono essere vere. Sono due
    controlli distinti apposta — uno protegge da un'attivazione accidentale della
    funzione, l'altro da un click distratto.
    """
    if not enabled:
        return AttachResult(False, (
            "l'iniezione a caldo e' disattivata. Puo' destabilizzare il processo "
            "target: abilitarla in configurazione (enable_hot_attach) solo se si "
            "accetta il rischio. In alternativa, avviare il processo con "
            "VRAMEXP_AGENT=1, che e' sicuro."), pid=pid)
    if not confirmed:
        return AttachResult(False, "serve una conferma esplicita per iniettare l'agent",
                            pid=pid)

    eligible, reason = check_eligibility(pid)
    if not eligible:
        return AttachResult(False, reason, pid=pid)

    target = _find_python_dll(pid)
    if target is None:
        return AttachResult(False, "DLL di CPython non piu' individuabile", pid=pid)

    try:
        addresses = _resolve_remote_symbols(target)
    except OSError as e:
        return AttachResult(False, f"risoluzione dei simboli fallita: {e}", pid=pid)

    code = _bootstrap_code()
    try:
        ok, detail = _inject(pid, addresses, code, timeout_s)
    except OSError as e:
        return AttachResult(False, f"iniezione fallita: {e}", pid=pid)
    if not ok:
        return AttachResult(False, detail, pid=pid)

    # Verifica dal lato client: l'iniezione puo' riuscire tecnicamente mentre il
    # codice Python fallisce in silenzio (import mancante, sys.path diverso).
    client = AgentClient()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.has_agent(pid):
            return AttachResult(True, "agent iniettato e operativo", pid=pid, detail=detail)
        time.sleep(0.3)
    return AttachResult(False, (
        "il codice e' stato iniettato ma l'agent non risponde: probabilmente il "
        "processo target non riesce a importare vramexp (sys.path diverso)"),
        pid=pid, detail=detail)


# -- individuazione del modulo CPython -------------------------------------
def _find_python_dll(pid: int) -> _Target | None:
    """Trova pythonXY.dll nel processo e la sua base remota."""
    try:
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:
        return None

    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return None
    try:
        needed = wintypes.DWORD()
        arr = (ctypes.c_void_p * 1024)()
        if not psapi.EnumProcessModulesEx(wintypes.HANDLE(handle), arr,
                                          ctypes.sizeof(arr), ctypes.byref(needed), 0x03):
            return None
        count = min(needed.value // ctypes.sizeof(ctypes.c_void_p), 1024)
        buf = ctypes.create_unicode_buffer(1024)
        for i in range(count):
            base = arr[i]
            if not base:
                continue
            if not psapi.GetModuleFileNameExW(wintypes.HANDLE(handle),
                                              ctypes.c_void_p(base), buf, 1024):
                continue
            path = buf.value
            if _PY_DLL_RE.match(os.path.basename(path)):
                return _Target(pid=pid, python_dll_path=path, remote_base=int(base))
        return None
    finally:
        kernel32.CloseHandle(handle)


def _bitness_matches(dll_path: str) -> bool:
    """Confronta l'architettura della DLL con quella dell'interprete corrente.

    Iniettare codice x64 in un processo a 32 bit (o viceversa) produce un crash
    immediato: e' un controllo che vale la pena fare prima, non dopo.
    """
    try:
        with open(dll_path, "rb") as f:
            if f.read(2) != b"MZ":
                return False
            f.seek(0x3C)
            pe_offset = struct.unpack("<I", f.read(4))[0]
            f.seek(pe_offset)
            if f.read(4) != b"PE\0\0":
                return False
            machine = struct.unpack("<H", f.read(2))[0]
    except OSError:
        return False
    is_64 = machine == 0x8664
    return is_64 == (struct.calcsize("P") == 8)


def _resolve_remote_symbols(target: _Target) -> dict[str, int]:
    """Indirizzi remoti di PyGILState_Ensure/PyRun_SimpleString/PyGILState_Release.

    Si carica la stessa DLL localmente per leggerne gli offset: le funzioni hanno
    lo stesso RVA in ogni processo che carica quel file, indipendentemente da
    dove l'ASLR abbia collocato il modulo.
    """
    local = ctypes.WinDLL(target.python_dll_path)
    local_base = ctypes.cast(local._handle, ctypes.c_void_p).value or 0
    if not local_base:
        raise OSError("base locale del modulo non determinabile")

    out: dict[str, int] = {}
    for symbol in ("PyGILState_Ensure", "PyRun_SimpleString", "PyGILState_Release"):
        try:
            addr = ctypes.cast(getattr(local, symbol), ctypes.c_void_p).value
        except AttributeError as e:
            raise OSError(f"simbolo {symbol} assente in {target.python_dll_path}") from e
        if not addr:
            raise OSError(f"indirizzo nullo per {symbol}")
        out[symbol] = target.remote_base + (addr - local_base)
    return out


def _bootstrap_code() -> bytes:
    """Codice Python eseguito nel processo target.

    Deve essere autosufficiente e non sollevare mai: un'eccezione non catturata
    dentro `PyRun_SimpleString` stampa un traceback sullo stdout del programma
    ospite, che nel migliore dei casi confonde e nel peggiore rompe un parser a
    valle. Il percorso del pacchetto viene passato esplicitamente perche' il
    target puo' avere un `sys.path` completamente diverso.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src = f"""
try:
    import sys
    _p = {repo_root!r}
    if _p not in sys.path:
        sys.path.insert(0, _p)
    import vramexp.probe.agent as _a
    _a.start()
except Exception:
    pass
"""
    return src.encode("utf-8") + b"\0"


def _inject(pid: int, addresses: dict[str, int], code: bytes,
            timeout_s: float) -> tuple[bool, str]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAllocEx.restype = ctypes.c_void_p
    kernel32.VirtualAllocEx.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t,
                                        wintypes.DWORD, wintypes.DWORD]
    kernel32.WriteProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    kernel32.CreateRemoteThread.restype = wintypes.HANDLE
    kernel32.CreateRemoteThread.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t,
                                            ctypes.c_void_p, ctypes.c_void_p,
                                            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]

    handle = kernel32.OpenProcess(INJECT_ACCESS, False, pid)
    if not handle:
        raise OSError(f"OpenProcess fallita: {ctypes.get_last_error()}")

    param_block = None
    code_block = None
    thread = None
    try:
        # Blocco parametri: i tre puntatori, lo scratch per il GIL, poi il sorgente.
        payload = struct.pack(
            "<QQQQ",
            addresses["PyGILState_Ensure"],
            addresses["PyRun_SimpleString"],
            addresses["PyGILState_Release"],
            0,
        ) + code
        param_block = kernel32.VirtualAllocEx(handle, None, len(payload),
                                              MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not param_block:
            raise OSError(f"VirtualAllocEx (parametri) fallita: {ctypes.get_last_error()}")
        written = ctypes.c_size_t(0)
        if not kernel32.WriteProcessMemory(handle, ctypes.c_void_p(param_block), payload,
                                           len(payload), ctypes.byref(written)):
            raise OSError(f"WriteProcessMemory (parametri) fallita: {ctypes.get_last_error()}")

        code_block = kernel32.VirtualAllocEx(handle, None, len(SHELLCODE),
                                             MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not code_block:
            raise OSError(f"VirtualAllocEx (codice) fallita: {ctypes.get_last_error()}")
        if not kernel32.WriteProcessMemory(handle, ctypes.c_void_p(code_block), SHELLCODE,
                                           len(SHELLCODE), ctypes.byref(written)):
            raise OSError(f"WriteProcessMemory (codice) fallita: {ctypes.get_last_error()}")

        # La pagina diventa eseguibile solo ora e senza permesso di scrittura:
        # scrivere ed eseguire contemporaneamente farebbe scattare le protezioni
        # antimalware su molte configurazioni.
        old = wintypes.DWORD()
        if not kernel32.VirtualProtectEx(handle, ctypes.c_void_p(code_block),
                                         ctypes.c_size_t(len(SHELLCODE)),
                                         PAGE_EXECUTE_READ, ctypes.byref(old)):
            raise OSError(f"VirtualProtectEx fallita: {ctypes.get_last_error()}")

        thread = kernel32.CreateRemoteThread(handle, None, 0,
                                             ctypes.c_void_p(code_block),
                                             ctypes.c_void_p(param_block), 0, None)
        if not thread:
            raise OSError(f"CreateRemoteThread fallita: {ctypes.get_last_error()}")

        rc = kernel32.WaitForSingleObject(thread, int(timeout_s * 1000))
        if rc == WAIT_TIMEOUT:
            # Non si libera la memoria remota: il thread potrebbe ancora usarla e
            # rilasciarla adesso provocherebbe un crash certo del target. Poche
            # decine di kilobyte trapelati sono un prezzo accettabile.
            param_block = code_block = None
            return False, ("il thread remoto non ha terminato entro il tempo previsto; "
                           "il processo target potrebbe avere il GIL bloccato")
        return True, f"thread remoto completato (codice {rc})"
    finally:
        if thread:
            kernel32.CloseHandle(thread)
        for block in (param_block, code_block):
            if block:
                kernel32.VirtualFreeEx(handle, ctypes.c_void_p(block), 0, MEM_RELEASE)
        kernel32.CloseHandle(handle)
