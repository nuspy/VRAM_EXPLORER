"""Riserva di VRAM tramite allocazione trattenuta ("balloon").

Windows non offre un modo per dire "tieni N GB liberi per me". L'unico modo
efficace di riservare VRAM e' **occuparla**: un processo dedicato alloca la
quantita' desiderata e la trattiene, cosi' gli altri client trovano meno memoria
disponibile e si adattano (o falliscono presto, invece di prendersi tutto e
lasciare a secco il lavoro che conta).

Onesta' sui limiti — la GUI deve dirlo, non nasconderlo:

* La riserva e' **efficace contro i client CUDA**: vedranno meno memoria libera e
  dimensioneranno di conseguenza (o falliranno subito).
* E' **debole contro il compositor e le applicazioni DXGI**: Windows puo'
  comunque sfrattare pagine per servirle. Non e' un blocco hardware.
* Riservare non rende la memoria piu' veloce ne' evita la frammentazione.

Implementazione con `nvcuda.dll` via ctypes invece che con torch: l'avvio e'
immediato e il processo occupa pochi megabyte di RAM, mentre importare torch
costerebbe secondi e centinaia di megabyte solo per fare `cudaMalloc`.

L'allocazione e' spezzata in blocchi (256 MB di default) per poter ridurre la
riserva gradualmente senza rilasciarla tutta e doverla riprendere.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

log = logging.getLogger("vramexp.balloon")

CUDA_SUCCESS = 0
CUDA_ERROR_OUT_OF_MEMORY = 2


class CudaError(RuntimeError):
    def __init__(self, func: str, code: int, message: str | None = None) -> None:
        self.code = code
        super().__init__(f"{func} fallita (codice {code}{': ' + message if message else ''})")


@dataclass
class BalloonState:
    requested_bytes: int = 0
    allocated_bytes: int = 0
    chunk_bytes: int = 0
    chunks: int = 0
    device: int = 0
    context_overhead_bytes: int = 0
    error: str | None = None

    @property
    def total_footprint_bytes(self) -> int:
        """VRAM realmente sottratta al sistema: riserva piu' contesto CUDA."""
        return self.allocated_bytes + self.context_overhead_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_bytes": self.requested_bytes,
            "allocated_bytes": self.allocated_bytes,
            "context_overhead_bytes": self.context_overhead_bytes,
            "total_footprint_bytes": self.total_footprint_bytes,
            "chunk_bytes": self.chunk_bytes,
            "chunks": self.chunks,
            "device": self.device,
            "error": self.error,
            "note": ("riserva best-effort: efficace contro i client CUDA, "
                     "non impedisce al compositor di ottenere memoria"),
        }


class VramBalloon:
    """Detiene blocchi di VRAM finche' non vengono rilasciati esplicitamente.

    Va usato **in-process**: il contesto CUDA appartiene al processo che lo crea,
    e la memoria viene liberata dal sistema alla sua morte. Il servizio lo avvia
    quindi in un processo figlio dedicato (`balloon_main.py`), cosi' un riavvio
    del servizio non trascina con se' la riserva se non lo si vuole.
    """

    def __init__(self, device: int = 0, chunk_bytes: int = 256 * 1024 * 1024) -> None:
        self.device = device
        self.chunk_bytes = max(16 * 1024 * 1024, chunk_bytes)
        self._blocks: list[ctypes.c_void_p] = []
        self._ctx: ctypes.c_void_p | None = None
        self._cuda = None
        self.error: str | None = None
        # Il solo contesto CUDA occupa VRAM prima di qualunque allocazione (sulla
        # RTX 5090 con driver 610 sono circa 2,7 GB). Va misurato e riportato,
        # altrimenti l'utente vede sparire piu' memoria di quanta ne ha chiesta e
        # pensa a un difetto.
        self.context_overhead_bytes = 0
        self._load()

    # -- inizializzazione --------------------------------------------------
    def _load(self) -> None:
        try:
            self._cuda = ctypes.WinDLL("nvcuda.dll")
        except OSError as e:
            self.error = f"nvcuda.dll non disponibile: {e}"
            return
        try:
            self._check("cuInit", self._cuda.cuInit(0))
            dev = ctypes.c_int()
            self._check("cuDeviceGet", self._cuda.cuDeviceGet(ctypes.byref(dev), self.device))
            # Entrambe le misure vengono da NVML: confrontare NVML con
            # cuMemGetInfo darebbe una differenza priva di senso, perche' le due
            # API contano in modo leggermente diverso e lo scarto fra loro
            # sommerebbe rumore all'overhead che si vuole misurare.
            free_before, _ = _nvml_free_total()
            ctx = ctypes.c_void_p()
            self._check("cuCtxCreate_v2",
                        self._cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
            self._ctx = ctx
            free_after, _ = _nvml_free_total()
            if free_before and free_after:
                self.context_overhead_bytes = max(0, free_before - free_after)
        except CudaError as e:
            self.error = str(e)

    @property
    def available(self) -> bool:
        return self._ctx is not None and self.error is None

    def _check(self, func: str, code: int) -> None:
        if code != CUDA_SUCCESS:
            raise CudaError(func, code, self._error_string(code))

    def _error_string(self, code: int) -> str | None:
        if self._cuda is None:
            return None
        try:
            buf = ctypes.c_char_p()
            if self._cuda.cuGetErrorString(code, ctypes.byref(buf)) == CUDA_SUCCESS and buf.value:
                return buf.value.decode("ascii", errors="replace")
        except Exception:
            pass
        return None

    # -- gestione della riserva -------------------------------------------
    @property
    def allocated_bytes(self) -> int:
        return len(self._blocks) * self.chunk_bytes

    def state(self, requested: int = 0) -> BalloonState:
        return BalloonState(
            requested_bytes=requested or self.allocated_bytes,
            allocated_bytes=self.allocated_bytes,
            chunk_bytes=self.chunk_bytes,
            chunks=len(self._blocks),
            device=self.device,
            context_overhead_bytes=self.context_overhead_bytes,
            error=self.error,
        )

    def resize(self, target_bytes: int) -> BalloonState:
        """Porta la riserva alla dimensione richiesta, in su o in giu'.

        Se la VRAM disponibile non basta, alloca quanto riesce e riporta nello
        stato la quantita' effettiva: fallire del tutto perche' mancano 200 MB
        sarebbe meno utile che riservare quello che c'e' e dirlo.
        """
        if not self.available:
            return BalloonState(requested_bytes=target_bytes, error=self.error or
                                "contesto CUDA non disponibile")
        target_chunks = max(0, int(target_bytes // self.chunk_bytes))

        while len(self._blocks) > target_chunks:
            self._free_one()

        partial_error: str | None = None
        while len(self._blocks) < target_chunks:
            try:
                self._alloc_one()
            except CudaError as e:
                partial_error = (
                    f"riservati {self.allocated_bytes / 2**30:.1f} GB dei "
                    f"{target_bytes / 2**30:.1f} GB richiesti: memoria insufficiente")
                log.info("balloon: %s (%s)", partial_error, e)
                break

        st = self.state(requested=target_bytes)
        st.error = partial_error
        return st

    def _alloc_one(self) -> None:
        ptr = ctypes.c_void_p()
        code = self._cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(self.chunk_bytes))
        if code != CUDA_SUCCESS:
            raise CudaError("cuMemAlloc_v2", code, self._error_string(code))
        self._blocks.append(ptr)

    def _free_one(self) -> None:
        ptr = self._blocks.pop()
        code = self._cuda.cuMemFree_v2(ptr)
        if code != CUDA_SUCCESS:
            # Non si rimette il blocco nella lista: il puntatore potrebbe essere
            # gia' invalido e ritentare all'infinito non aiuterebbe.
            log.warning("cuMemFree fallita (codice %s): blocco abbandonato", code)

    def release_all(self) -> None:
        while self._blocks:
            self._free_one()

    def close(self) -> None:
        self.release_all()
        if self._ctx is not None and self._cuda is not None:
            try:
                self._cuda.cuCtxDestroy_v2(self._ctx)
            except Exception:
                pass
            self._ctx = None

    def __enter__(self) -> "VramBalloon":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _nvml_free_total(device: int = 0) -> tuple[int, int]:
    """(libera, totale) secondo NVML.

    NVML non richiede un contesto CUDA corrente, quindi funziona anche prima che
    il balloon ne abbia creato uno: e' cio' che permette di misurare quanto costi
    il contesto stesso. Un fallimento e' silenzioso perche' questa e' una misura
    accessoria, non un requisito per riservare memoria.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(m.free), int(m.total)
    except Exception:
        return (0, 0)


def free_vram_bytes(device: int = 0) -> tuple[int, int]:
    """(libera, totale) secondo il driver CUDA.

    Utile prima di dimensionare la riserva: e' la stessa cifra che vedra' il
    prossimo processo CUDA che prova ad allocare.
    """
    try:
        cuda = ctypes.WinDLL("nvcuda.dll")
        if cuda.cuInit(0) != CUDA_SUCCESS:
            return (0, 0)
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), device) != CUDA_SUCCESS:
            return (0, 0)
        ctx = ctypes.c_void_p()
        if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != CUDA_SUCCESS:
            return (0, 0)
        try:
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != CUDA_SUCCESS:
                return (0, 0)
            return (int(free.value), int(total.value))
        finally:
            cuda.cuCtxDestroy_v2(ctx)
    except Exception:
        return (0, 0)
