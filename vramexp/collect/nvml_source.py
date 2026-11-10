"""Telemetria delle GPU NVIDIA via NVML.

NVML e' la fonte autorevole per lo stato **complessivo** della GPU: memoria
residente, temperature, clock, potenza, throttling, BAR1. Non lo e' per la
ripartizione per processo: su driver WDDM (tutte le GeForce su Windows client)
`usedGpuMemory` e' sempre `None`. Quella parte arriva da `pdh_source`.

Ogni lettura opzionale e' protetta singolarmente: su GeForce diverse query
datacenter (ECC, persistence mode, accounting) rispondono `NotSupported`, ed e'
un esito atteso, non un errore da propagare.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

try:
    import pynvml as nvml
    _NVML_IMPORT_ERROR: str | None = None
except ImportError as e:                                  # pragma: no cover
    nvml = None                                           # type: ignore[assignment]
    _NVML_IMPORT_ERROR = str(e)

from ..core.models import GpuInfo, GpuSample

T = TypeVar("T")

_ARCH_NAMES = {
    2: "Kepler", 3: "Maxwell", 4: "Pascal", 5: "Volta", 6: "Turing",
    7: "Ampere", 8: "Ada", 9: "Hopper", 10: "Blackwell", 11: "Rubin",
}

THROTTLE_FLAGS = [
    (0x0000000000000001, "GPU idle"),
    (0x0000000000000002, "limite clock applicazione"),
    (0x0000000000000004, "limite software power cap"),
    (0x0000000000000008, "HW slowdown"),
    (0x0000000000000010, "sincronizzazione boost"),
    (0x0000000000000020, "limite software termico"),
    (0x0000000000000040, "HW slowdown termico"),
    (0x0000000000000080, "HW slowdown alimentazione"),
    (0x0000000000000100, "limite display clock"),
]


def describe_throttle(mask: int | None) -> list[str]:
    if not mask:
        return []
    return [label for flag, label in THROTTLE_FLAGS if mask & flag]


def _try(fn: Callable[[], T], default: T | None = None) -> T | None:
    """Esegue una query NVML tollerando il fallimento.

    Serve perche' il supporto varia per modello e driver: la strategia e'
    chiedere tutto e tenere cio' che risponde, invece di mantenere una matrice di
    compatibilita' che invecchierebbe a ogni release del driver.
    """
    try:
        return fn()
    except Exception:
        return default


class NvmlSource:
    """Accesso NVML con inizializzazione pigra e thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handles: list[Any] = []
        self._info: list[GpuInfo] = []
        self.available = False
        self.init_error: str | None = _NVML_IMPORT_ERROR
        self._init()

    def _init(self) -> None:
        if nvml is None:
            return
        try:
            nvml.nvmlInit()
        except Exception as e:
            self.init_error = f"nvmlInit fallita: {e}"
            return

        driver = _try(lambda: _decode(nvml.nvmlSystemGetDriverVersion()))
        count = _try(lambda: nvml.nvmlDeviceGetCount(), 0) or 0
        for i in range(count):
            h = _try(lambda i=i: nvml.nvmlDeviceGetHandleByIndex(i))
            if h is None:
                continue
            self._handles.append(h)
            mem = _try(lambda h=h: nvml.nvmlDeviceGetMemoryInfo(h))
            arch = _try(lambda h=h: nvml.nvmlDeviceGetArchitecture(h))
            pci = _try(lambda h=h: nvml.nvmlDeviceGetPciInfo(h))
            self._info.append(GpuInfo(
                index=i,
                name=_decode(_try(lambda h=h: nvml.nvmlDeviceGetName(h)) or "GPU sconosciuta"),
                uuid=_decode(_try(lambda h=h: nvml.nvmlDeviceGetUUID(h))),
                pci_bus_id=_decode(getattr(pci, "busId", None)) if pci else None,
                total_bytes=int(getattr(mem, "total", 0) or 0),
                architecture=_ARCH_NAMES.get(arch, f"arch{arch}") if arch is not None else None,
                cuda_cores=_try(lambda h=h: nvml.nvmlDeviceGetNumGpuCores(h)),
                memory_bus_width=_try(lambda h=h: nvml.nvmlDeviceGetMemoryBusWidth(h)),
                driver_version=driver,
                vbios=_decode(_try(lambda h=h: nvml.nvmlDeviceGetVbiosVersion(h))),
            ))
        self.available = bool(self._handles)
        if not self.available and not self.init_error:
            self.init_error = "nessuna GPU NVIDIA rilevata da NVML"

    # -- accesso -----------------------------------------------------------
    @property
    def gpu_info(self) -> list[GpuInfo]:
        return list(self._info)

    def identity_tuples(self) -> list[tuple[int, str, int]]:
        """(index, name, total_bytes) per la mappatura LUID di `dxgi`."""
        return [(g.index, g.name, g.total_bytes) for g in self._info]

    def sample(self) -> list[GpuSample]:
        if not self.available:
            return []
        out: list[GpuSample] = []
        with self._lock:
            for i, h in enumerate(self._handles):
                out.append(self._sample_one(i, h))
        return out

    def _sample_one(self, index: int, h: Any) -> GpuSample:
        s = GpuSample(gpu_index=index)

        mem = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(h, version=2))
        if mem is None:
            mem = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(h))
        if mem is not None:
            s.total_bytes = int(getattr(mem, "total", 0) or 0)
            s.used_bytes = int(getattr(mem, "used", 0) or 0)
            s.free_bytes = int(getattr(mem, "free", 0) or 0)
            s.reserved_bytes = int(getattr(mem, "reserved", 0) or 0)

        bar1 = _try(lambda: nvml.nvmlDeviceGetBAR1MemoryInfo(h))
        if bar1 is not None:
            s.bar1_total_bytes = int(bar1.bar1Total)
            s.bar1_used_bytes = int(bar1.bar1Used)

        util = _try(lambda: nvml.nvmlDeviceGetUtilizationRates(h))
        if util is not None:
            s.util_gpu_pct = int(util.gpu)
            s.util_memory_pct = int(util.memory)

        enc = _try(lambda: nvml.nvmlDeviceGetEncoderUtilization(h))
        if enc:
            s.util_encoder_pct = int(enc[0])
        dec = _try(lambda: nvml.nvmlDeviceGetDecoderUtilization(h))
        if dec:
            s.util_decoder_pct = int(dec[0])

        s.temperature_c = _try(lambda: nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU))
        power = _try(lambda: nvml.nvmlDeviceGetPowerUsage(h))
        s.power_w = power / 1000.0 if power is not None else None
        limit = _try(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(h))
        s.power_limit_w = limit / 1000.0 if limit is not None else None
        s.fan_pct = _try(lambda: nvml.nvmlDeviceGetFanSpeed(h))
        s.clock_sm_mhz = _try(lambda: nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM))
        s.clock_mem_mhz = _try(lambda: nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM))
        s.throttle_reasons = _try(
            lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h))
        s.pcie_tx_kbs = _try(lambda: nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_TX_BYTES))
        s.pcie_rx_kbs = _try(lambda: nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_RX_BYTES))
        return s

    def running_pids(self, index: int) -> set[int]:
        """PID che hanno un contesto attivo sulla GPU.

        La memoria associata non e' leggibile su WDDM, ma il solo elenco dei PID
        e' comunque utile: distingue chi usa davvero la GPU da chi ha soltanto
        una superficie di composizione, e conferma i dati PDH.
        """
        if not self.available or index >= len(self._handles):
            return set()
        h = self._handles[index]
        pids: set[int] = set()
        for fn in (getattr(nvml, "nvmlDeviceGetComputeRunningProcesses", None),
                   getattr(nvml, "nvmlDeviceGetGraphicsRunningProcesses", None)):
            if fn is None:
                continue
            procs = _try(lambda fn=fn: fn(h), []) or []
            pids.update(int(p.pid) for p in procs)
        return pids

    def shutdown(self) -> None:
        if self.available and nvml is not None:
            _try(lambda: nvml.nvmlShutdown())
            self.available = False


def _decode(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)
