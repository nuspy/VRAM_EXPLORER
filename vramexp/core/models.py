"""Modelli dati condivisi fra servizio, API, CLI e GUI.

Sono dataclass semplici (non pydantic) perche' vengono create ~1 volta al secondo
per ogni processo: la validazione runtime di pydantic sarebbe un costo inutile su
dati che produciamo noi stessi. La serializzazione verso l'API passa da `to_dict`.

Convenzione: tutte le dimensioni di memoria sono in **byte**, tutti i timestamp
sono epoch UTC in secondi (float).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Quanto ci fidiamo dell'identificazione del contenuto.

    Esiste perche' la GUI non deve mai presentare un'inferenza euristica con la
    stessa autorevolezza di un dato letto dall'API del backend.
    """

    CONFIRMED = "confirmed"   # risposta diretta del processo o del suo backend
    PARSED = "parsed"         # letto da cmdline o da header di un file su disco
    INFERRED = "inferred"     # euristica su nome eseguibile / categoria nota
    UNKNOWN = "unknown"


class ActionKind(str, Enum):
    """Azioni di rilascio VRAM, in ordine crescente di invasivita'."""

    UNLOAD_MODEL = "unload_model"     # API del backend: pulita, processo vivo
    EMPTY_CACHE = "empty_cache"       # torch.cuda.empty_cache() via agent
    TRIM = "trim"                     # EmptyWorkingSet
    SUSPEND = "suspend"               # NtSuspendProcess
    RESUME = "resume"
    KILL = "kill"                     # ultima risorsa


@dataclass(slots=True)
class GpuInfo:
    """Identita' statica di una GPU: non cambia fra un campione e l'altro."""

    index: int
    name: str
    uuid: str | None = None
    pci_bus_id: str | None = None
    luid: str | None = None          # chiave di join con le istanze PDH
    total_bytes: int = 0
    architecture: str | None = None
    cuda_cores: int | None = None
    memory_bus_width: int | None = None
    driver_version: str | None = None
    vbios: str | None = None
    is_integrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GpuSample:
    """Stato istantaneo di una GPU.

    Nota su `used_bytes` vs `committed_bytes`: il primo viene da NVML/adapter ed
    e' la memoria **residente** in VRAM fisica; il secondo e' il totale
    *committed* contato da Windows, che puo' superare la capacita' fisica perche'
    l'OS fa overcommit e sposta pagine verso la RAM. Presentarli come se fossero
    la stessa cosa e' l'errore piu' facile da commettere in questo dominio.
    """

    gpu_index: int
    ts: float = field(default_factory=time.time)

    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    reserved_bytes: int = 0

    committed_bytes: int | None = None      # GPU Adapter Memory\Total Committed
    shared_used_bytes: int | None = None    # sysmem usata dall'adapter

    bar1_total_bytes: int | None = None
    bar1_used_bytes: int | None = None

    util_gpu_pct: int | None = None
    util_memory_pct: int | None = None      # utilizzo del memory controller
    util_encoder_pct: int | None = None
    util_decoder_pct: int | None = None

    temperature_c: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: int | None = None
    clock_sm_mhz: int | None = None
    clock_mem_mhz: int | None = None
    throttle_reasons: int | None = None
    pcie_tx_kbs: int | None = None
    pcie_rx_kbs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContentInfo:
    """Cosa c'e' dentro la VRAM di un processo, per quanto siamo riusciti a capirlo."""

    kind: str = "unknown"             # llm | vlm | embeddings | diffusion | compositor | browser | game | ...
    label: str = ""                   # testo breve per la GUI
    detail: str = ""                  # riga secondaria
    confidence: Confidence = Confidence.UNKNOWN
    source: str = ""                  # quale resolver l'ha prodotto
    model_name: str | None = None
    quantization: str | None = None
    parameters: str | None = None     # "35B", "7B", ...
    context_length: int | None = None
    file_path: str | None = None
    file_size_bytes: int | None = None
    backend: str | None = None        # llama.cpp | lmstudio | comfyui | ...
    endpoint: str | None = None       # per l'unload mirato
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence.value
        return d


@dataclass(slots=True)
class ProcSample:
    """Consumo VRAM di un singolo processo su una singola GPU.

    `dedicated_bytes` e' committed per-processo (somma > VRAM fisica e' normale).
    `shared_bytes > 0` significa che il processo sta usando RAM di sistema al
    posto della VRAM: e' il segnale di spillover, con penalita' di 10-100x.
    """

    pid: int
    gpu_index: int
    ts: float = field(default_factory=time.time)

    dedicated_bytes: int = 0
    shared_bytes: int = 0
    committed_bytes: int = 0
    local_bytes: int = 0
    non_local_bytes: int = 0

    engine_pct: dict[str, float] = field(default_factory=dict)  # 3D, Compute, Copy, VideoDecode, ...

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcInfo:
    """Identita' di un processo: costosa da raccogliere, quindi memorizzata a parte
    dai campioni e aggiornata solo quando il PID compare per la prima volta."""

    pid: int
    name: str = ""
    exe: str | None = None
    cmdline: str = ""
    username: str | None = None
    publisher: str | None = None
    create_time: float | None = None
    icon_png_b64: str | None = None
    content: ContentInfo = field(default_factory=ContentInfo)
    available_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content"] = self.content.to_dict()
        return d


@dataclass(slots=True)
class ProcessEntry:
    """Vista unita' processo + campione, cosi' come la consumano GUI e API."""

    info: ProcInfo
    sample: ProcSample

    def to_dict(self) -> dict[str, Any]:
        d = self.info.to_dict()
        d.update(self.sample.to_dict())
        return d


@dataclass(slots=True)
class Snapshot:
    """Fotografia completa del sistema in un istante."""

    ts: float = field(default_factory=time.time)
    gpus: list[GpuSample] = field(default_factory=list)
    gpu_info: list[GpuInfo] = field(default_factory=list)
    processes: list[ProcessEntry] = field(default_factory=list)
    balloon_bytes: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "gpus": [g.to_dict() for g in self.gpus],
            "gpu_info": [g.to_dict() for g in self.gpu_info],
            "processes": [p.to_dict() for p in self.processes],
            "balloon_bytes": self.balloon_bytes,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class Incident:
    """Evento degno di autopsia: sparizione improvvisa di un processo sotto
    pressione di memoria, oppure inizio di spillover su RAM."""

    ts: float
    kind: str                     # "oom_suspect" | "spillover" | "process_gone"
    pid: int
    process_name: str
    content_label: str
    gpu_index: int
    peak_dedicated_bytes: int
    gpu_used_bytes: int
    gpu_total_bytes: int
    detail: str = ""
    history_json: str = ""        # i 60 s precedenti, serializzati
    id: int = 0                   # assegnato dallo store al salvataggio

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
