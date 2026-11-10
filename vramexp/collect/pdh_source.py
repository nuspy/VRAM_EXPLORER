"""Lettura dei performance counter GPU di Windows tramite PDH nativo.

**Perche' PDH e non NVML.** Su GeForce con driver WDDM, NVML rifiuta di riportare
la memoria per-processo (`nvmlDeviceGetComputeRunningProcesses` restituisce
`usedGpuMemory = None`, e `nvidia-smi` stampa "Not available in WDDM driver
model"). I performance counter `GPU Process Memory` invece funzionano: sono la
stessa fonte che alimenta la scheda Prestazioni del Task Manager. Sono quindi
l'unico modo su questa piattaforma per sapere *chi* occupa la VRAM.

**Perche' ctypes e non `Get-Counter`/WMI.** Una query PDH persistente costa
frazioni di millisecondo per campione; lanciare PowerShell a 1 Hz costerebbe
centinaia di millisecondi e un processo nuovo ogni volta.

**Nomi dei contatori.** Si usa `PdhAddEnglishCounterW`, che accetta i nomi
inglesi anche su Windows localizzato: su un sistema in italiano i nomi tradotti
farebbero fallire l'aggiunta del contatore.

Formato delle istanze prodotte da Windows:
    GPU Process Memory : pid_<PID>_luid_0x<HI>_0x<LO>_phys_<N>
    GPU Adapter Memory :         luid_0x<HI>_0x<LO>_phys_<N>
    GPU Engine         : pid_<PID>_luid_..._phys_<N>_eng_<N>_engtype_<TIPO>
Il LUID identifica l'adapter (necessario in multi-GPU), `phys_<N>` il segmento di
memoria: un processo puo' comparire in piu' istanze, che vanno sommate.
"""

from __future__ import annotations

import ctypes
import re
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Iterator

_pdh = ctypes.WinDLL("pdh.dll")

# --- costanti PDH ---------------------------------------------------------
PDH_FMT_LARGE = 0x00000400
PDH_FMT_DOUBLE = 0x00000200
PDH_FMT_NOCAP100 = 0x00008000

PDH_CSTATUS_VALID_DATA = 0x00000000
PDH_CSTATUS_NEW_DATA = 0x00000001
PDH_MORE_DATA = 0x800007D2
PDH_NO_DATA = 0x800007D5
PDH_INVALID_DATA = 0xC0000BC6
PDH_INVALID_HANDLE = 0xC0000BBC
PDH_CSTATUS_NO_INSTANCE = 0x800007D1
PDH_CSTATUS_NO_COUNTER = 0xC0000BB9
PDH_CSTATUS_NO_OBJECT = 0xC0000BB8


class _PDH_COUNTERVALUE_LARGE(ctypes.Structure):
    # CStatus e' DWORD, largeValue e' LONGLONG: ctypes inserisce da solo i 4 byte
    # di padding richiesti dall'allineamento a 8 su x64.
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _PDH_COUNTERVALUE_DOUBLE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _PDH_ITEM_LARGE(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _PDH_COUNTERVALUE_LARGE)]


class _PDH_ITEM_DOUBLE(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _PDH_COUNTERVALUE_DOUBLE)]


_pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_pdh.PdhOpenQueryW.restype = ctypes.c_long
_pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_void_p)]
_pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
_pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
_pdh.PdhCollectQueryData.restype = ctypes.c_long
_pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.DWORD),
                                              ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
_pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
_pdh.PdhCloseQuery.restype = ctypes.c_long
_pdh.PdhRemoveCounter.argtypes = [ctypes.c_void_p]
_pdh.PdhRemoveCounter.restype = ctypes.c_long


def _u32(status: int) -> int:
    """PDH restituisce codici negativi come c_long; servono come unsigned."""
    return status & 0xFFFFFFFF


# --- parsing dei nomi di istanza -----------------------------------------
_RE_PROC = re.compile(
    r"^pid_(?P<pid>\d+)_luid_(?P<hi>0x[0-9A-Fa-f]+)_(?P<lo>0x[0-9A-Fa-f]+)_phys_(?P<phys>\d+)",
    re.IGNORECASE,
)
_RE_ADAPTER = re.compile(
    r"^luid_(?P<hi>0x[0-9A-Fa-f]+)_(?P<lo>0x[0-9A-Fa-f]+)_phys_(?P<phys>\d+)",
    re.IGNORECASE,
)
_RE_ENGTYPE = re.compile(r"_engtype_(?P<engtype>.+)$", re.IGNORECASE)


def normalize_luid(hi: str, lo: str) -> str:
    """Forma canonica del LUID, per usarlo come chiave di dizionario.

    Windows scrive gli hex talvolta in minuscolo e talvolta in maiuscolo nella
    stessa sessione; senza normalizzazione i join fra contatori diversi
    fallirebbero in modo intermittente e difficile da diagnosticare.
    """
    return f"0x{int(hi, 16):08X}_0x{int(lo, 16):08X}"


def parse_process_instance(name: str) -> tuple[int, str] | None:
    m = _RE_PROC.match(name)
    if not m:
        return None
    return int(m.group("pid")), normalize_luid(m.group("hi"), m.group("lo"))


def parse_adapter_instance(name: str) -> str | None:
    m = _RE_ADAPTER.match(name)
    if not m:
        return None
    return normalize_luid(m.group("hi"), m.group("lo"))


def parse_engtype(name: str) -> str:
    m = _RE_ENGTYPE.search(name)
    return m.group("engtype") if m else "Unknown"


# --- contatori richiesti --------------------------------------------------
# chiave logica -> percorso PDH (nomi inglesi)
PROC_COUNTERS = {
    "dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
    "shared": r"\GPU Process Memory(*)\Shared Usage",
    "committed": r"\GPU Process Memory(*)\Total Committed",
    "local": r"\GPU Process Memory(*)\Local Usage",
    "non_local": r"\GPU Process Memory(*)\Non Local Usage",
}
ADAPTER_COUNTERS = {
    "dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
    "shared": r"\GPU Adapter Memory(*)\Shared Usage",
    "committed": r"\GPU Adapter Memory(*)\Total Committed",
}
ENGINE_COUNTER = r"\GPU Engine(*)\Utilization Percentage"


@dataclass
class ProcMem:
    """Somma dei segmenti fisici di un processo su un dato adapter."""

    pid: int
    luid: str
    dedicated: int = 0
    shared: int = 0
    committed: int = 0
    local: int = 0
    non_local: int = 0
    engines: dict[str, float] = field(default_factory=dict)


@dataclass
class AdapterMem:
    luid: str
    dedicated: int = 0
    shared: int = 0
    committed: int = 0


@dataclass
class PdhSnapshot:
    processes: dict[tuple[int, str], ProcMem] = field(default_factory=dict)
    adapters: dict[str, AdapterMem] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def by_pid(self) -> Iterator[ProcMem]:
        return iter(self.processes.values())


class PdhCollector:
    """Query PDH persistente sui contatori GPU.

    La query resta aperta per tutta la vita del servizio: riaprirla a ogni
    campione costerebbe piu' della lettura stessa. `sample()` non solleva mai per
    errori PDH transitori (un adapter che sparisce, un contatore momentaneamente
    senza istanze): restituisce uno snapshot con `ok=False`, perche' il ciclo di
    campionamento non deve morire per un buco di un secondo.
    """

    def __init__(self, with_engines: bool = True) -> None:
        self._query: ctypes.c_void_p | None = None
        self._proc_handles: dict[str, ctypes.c_void_p] = {}
        self._adapter_handles: dict[str, ctypes.c_void_p] = {}
        self._engine_handle: ctypes.c_void_p | None = None
        self._with_engines = with_engines
        self._primed = False          # i contatori rate richiedono due collect
        self.available = False
        self.init_error: str | None = None
        self._open()

    # -- ciclo di vita ----------------------------------------------------
    def _open(self) -> None:
        q = ctypes.c_void_p()
        st = _u32(_pdh.PdhOpenQueryW(None, None, ctypes.byref(q)))
        if st != 0:
            self.init_error = f"PdhOpenQuery fallita: 0x{st:08X}"
            return
        self._query = q

        for key, path in PROC_COUNTERS.items():
            h = self._add(path)
            if h is not None:
                self._proc_handles[key] = h
        for key, path in ADAPTER_COUNTERS.items():
            h = self._add(path)
            if h is not None:
                self._adapter_handles[key] = h
        if self._with_engines:
            self._engine_handle = self._add(ENGINE_COUNTER)

        if "dedicated" not in self._proc_handles:
            # Senza il contatore per-processo il servizio perde la sua ragione
            # d'essere: meglio dirlo subito e chiaramente.
            self.init_error = (
                "contatore 'GPU Process Memory\\Dedicated Usage' non disponibile. "
                "Verificare che il set di contatori GPU sia registrato "
                "(lodctr /R) e che l'utente appartenga a 'Performance Monitor Users'."
            )
            return
        self.available = True

    def _add(self, path: str) -> ctypes.c_void_p | None:
        h = ctypes.c_void_p()
        st = _u32(_pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(h)))
        if st != 0:
            return None
        return h

    def close(self) -> None:
        if self._query is not None:
            _pdh.PdhCloseQuery(self._query)
            self._query = None
        self.available = False

    def __enter__(self) -> "PdhCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- lettura ----------------------------------------------------------
    def _read_array(self, handle: ctypes.c_void_p, as_double: bool = False
                    ) -> list[tuple[str, float]]:
        """Legge un contatore con istanze wildcard.

        PDH vuole essere chiamato due volte: la prima con buffer nullo per farsi
        dire la dimensione (risponde PDH_MORE_DATA), la seconda con il buffer
        allocato.
        """
        item_t = _PDH_ITEM_DOUBLE if as_double else _PDH_ITEM_LARGE
        fmt = (PDH_FMT_DOUBLE | PDH_FMT_NOCAP100) if as_double else PDH_FMT_LARGE

        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        st = _u32(_pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size),
                                                    ctypes.byref(count), None))
        if st != PDH_MORE_DATA:
            # Nessuna istanza o dato non ancora valido: normale al primo giro e
            # quando nessun processo usa la GPU.
            return []
        if count.value == 0 or size.value == 0:
            return []

        buf = ctypes.create_string_buffer(size.value)
        st = _u32(_pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size),
                                                    ctypes.byref(count), buf))
        if st != 0:
            return []

        items = ctypes.cast(buf, ctypes.POINTER(item_t))
        out: list[tuple[str, float]] = []
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus not in (PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA):
                continue
            name = it.szName or ""
            value = it.FmtValue.doubleValue if as_double else it.FmtValue.largeValue
            out.append((name, float(value)))
        return out

    def sample(self) -> PdhSnapshot:
        snap = PdhSnapshot()
        if not self.available or self._query is None:
            snap.ok = False
            snap.error = self.init_error or "collector PDH non disponibile"
            return snap

        st = _u32(_pdh.PdhCollectQueryData(self._query))
        if st != 0 and st != PDH_NO_DATA:
            snap.ok = False
            snap.error = f"PdhCollectQueryData: 0x{st:08X}"
            return snap

        for key, handle in self._proc_handles.items():
            for name, value in self._read_array(handle):
                parsed = parse_process_instance(name)
                if parsed is None:
                    continue
                pid, luid = parsed
                pm = snap.processes.get((pid, luid))
                if pm is None:
                    pm = ProcMem(pid=pid, luid=luid)
                    snap.processes[(pid, luid)] = pm
                # I segmenti fisici dello stesso processo si sommano.
                setattr(pm, key, int(getattr(pm, key) + value))

        for key, handle in self._adapter_handles.items():
            for name, value in self._read_array(handle):
                luid = parse_adapter_instance(name)
                if luid is None:
                    continue
                am = snap.adapters.get(luid)
                if am is None:
                    am = AdapterMem(luid=luid)
                    snap.adapters[luid] = am
                setattr(am, key, int(getattr(am, key) + value))

        if self._engine_handle is not None and self._primed:
            # Utilization Percentage e' un contatore rate: al primo campione non
            # esiste un intervallo su cui calcolarlo, quindi si salta.
            for name, value in self._read_array(self._engine_handle, as_double=True):
                parsed = parse_process_instance(name)
                if parsed is None or value <= 0.0:
                    continue
                pid, luid = parsed
                pm = snap.processes.get((pid, luid))
                if pm is None:
                    pm = ProcMem(pid=pid, luid=luid)
                    snap.processes[(pid, luid)] = pm
                engtype = parse_engtype(name)
                pm.engines[engtype] = pm.engines.get(engtype, 0.0) + value

        self._primed = True
        return snap
