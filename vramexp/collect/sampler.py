"""Unione delle fonti in uno `Snapshot` coerente.

Il lavoro delicato di questo modulo e' **riconciliare due contabilita' diverse**:

  * NVML e `GPU Adapter Memory` riportano la memoria *residente* sulla GPU
    (25,8 GB su 31,8 nel caso di prova);
  * `GPU Process Memory\\Dedicated Usage` riporta il *committed* per processo, la
    cui somma puo' superare la VRAM fisica (39,9 GB misurati) perche' Windows fa
    overcommit e sposta pagine verso la RAM quando serve.

Nessuna delle due e' sbagliata: misurano cose diverse. Sommare i per-processo e
presentarli come occupazione reale sarebbe pero' una bugia, quindi lo `Snapshot`
espone entrambe le grandezze separatamente e la GUI usa il totale adapter come
denominatore. `scale_factor` fornisce inoltre una stima proporzionale della quota
residente di ciascun processo, esplicitamente etichettata come stima.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from ..core.config import Config
from ..core.models import GpuInfo, GpuSample, ProcessEntry, ProcInfo, ProcSample, Snapshot
from .dxgi import enum_adapters, map_luid_to_nvml
from .nvml_source import NvmlSource
from .pdh_source import PdhCollector, ProcMem
from .proc_source import ProcSource


class ContentResolver(Protocol):
    """Interfaccia minima implementata da `vramexp.content.registry`."""

    def resolve(self, info: ProcInfo) -> None: ...


class Sampler:
    """Produce uno `Snapshot` completo a ogni chiamata di `sample()`.

    Le sorgenti costose sono disaccoppiate dalla frequenza di campionamento:
    l'identita' dei processi e' in cache, e la risoluzione del contenuto (che
    interroga API HTTP) e' limitata da `content_refresh_s`.
    """

    def __init__(self, cfg: Config | None = None,
                 content_resolver: ContentResolver | None = None) -> None:
        self.cfg = cfg or Config()
        self.nvml = NvmlSource()
        self.pdh = PdhCollector(with_engines=True)
        self.procs = ProcSource(want_icons=True)
        self.content_resolver = content_resolver

        self._luid_to_gpu: dict[str, int] = {}
        self._gpu_to_luid: dict[int, str] = {}
        self._refresh_adapter_map()

        self._last_content_scan = 0.0
        self._balloon_bytes_provider: Callable[[], int] | None = None

    # -- configurazione ----------------------------------------------------
    def set_balloon_provider(self, fn: Callable[[], int]) -> None:
        self._balloon_bytes_provider = fn

    def _refresh_adapter_map(self) -> None:
        try:
            adapters = enum_adapters()
        except Exception:
            adapters = []
        self._luid_to_gpu = map_luid_to_nvml(adapters, self.nvml.identity_tuples())
        self._gpu_to_luid = {v: k for k, v in self._luid_to_gpu.items()}
        self._dxgi_adapters = {a.luid: a for a in adapters}

        # Aggiungi la GPU integrata se presente e non gia' mappata
        self._integrated_gpu_luid: str | None = None
        for a in adapters:
            if a.is_integrated and a.luid not in self._luid_to_gpu:
                self._integrated_gpu_luid = a.luid
                # Assegna alla GPU integrata l'indice -1 per distinguerla dalle NVIDIA
                self._luid_to_gpu[a.luid] = -1
                self._gpu_to_luid[-1] = a.luid
                break

    @property
    def gpu_info(self) -> list[GpuInfo]:
        info = self.nvml.gpu_info
        for g in info:
            g.luid = self._gpu_to_luid.get(g.index)

        # Aggiungi la GPU integrata se identificata
        if self._integrated_gpu_luid:
            from ..core.models import GpuInfo as GpuInfoClass
            integrated = GpuInfoClass(
                index=-1,
                name="GPU Integrata (Intel)",
                luid=self._integrated_gpu_luid,
                total_bytes=0,  # Verrà riempito da PDH
            )
            info.append(integrated)

        return info

    def health(self) -> dict[str, object]:
        """Diagnostica delle sorgenti, esposta dall'API: se un dato manca,
        l'utente deve poter capire quale collector non sta rispondendo."""
        return {
            "nvml_available": self.nvml.available,
            "nvml_error": self.nvml.init_error,
            "pdh_available": self.pdh.available,
            "pdh_error": self.pdh.init_error,
            "adapters_mapped": len(self._luid_to_gpu),
            "gpu_count": len(self.nvml.gpu_info),
        }

    # -- campionamento -----------------------------------------------------
    def sample(self) -> Snapshot:
        now = time.time()
        snap = Snapshot(ts=now)
        snap.gpu_info = self.gpu_info

        gpu_samples = self.nvml.sample()
        pdh_snap = self.pdh.sample()

        if not pdh_snap.ok and pdh_snap.error:
            snap.warnings.append(f"PDH: {pdh_snap.error}")
        if not self.nvml.available and self.nvml.init_error:
            snap.warnings.append(f"NVML: {self.nvml.init_error}")

        # Se PDH vede un LUID che non sappiamo mappare, un adapter e' comparso
        # (eGPU collegata, driver reinstallato): si rifa' la mappa una volta.
        unknown = set(pdh_snap.adapters) - set(self._luid_to_gpu) - set(self._dxgi_adapters)
        if unknown:
            self._refresh_adapter_map()

        # Totali per adapter: completano NVML con committed e shared, che NVML
        # non espone affatto.
        by_index: dict[int, GpuSample] = {g.gpu_index: g for g in gpu_samples}
        for luid, adapter in pdh_snap.adapters.items():
            idx = self._luid_to_gpu.get(luid)
            if idx is None:
                continue

            # Se è la GPU integrata (idx == -1) e non è ancora in by_index,
            # creala al volo
            if idx == -1 and idx not in by_index:
                g = GpuSample(gpu_index=-1)
                by_index[idx] = g
            elif idx not in by_index:
                continue

            g = by_index[idx]
            g.committed_bytes = adapter.committed
            g.shared_used_bytes = adapter.shared
            if g.total_bytes == 0 and adapter.dedicated:
                # GPU non NVIDIA o integrata: nessun dato NVML, si usa quel che da' PDH.
                g.used_bytes = adapter.dedicated
                g.total_bytes = adapter.dedicated

        snap.gpus = list(by_index.values())

        # Processi. Un PID puo' comparire su piu' adapter: ogni coppia
        # (pid, gpu) diventa una riga distinta, com'e' corretto in multi-GPU.
        do_content = (now - self._last_content_scan) >= self.cfg.content_refresh_s
        if do_content:
            self._last_content_scan = now

        live_pids: set[int] = set()
        min_bytes = self.cfg.min_track_bytes

        for (pid, luid), pm in pdh_snap.processes.items():
            gpu_index = self._luid_to_gpu.get(luid, -1)

            # Per la GPU integrata (gpu_index == -1), PDH non traccia memoria dedicata
            # (sempre 0), quindi includiamo i processi comunque. Per altre GPU,
            # applichiamo la soglia minima per filtrare il rumore.
            if gpu_index != -1 and pm.dedicated < min_bytes and pm.shared < min_bytes and pm.committed < min_bytes:
                continue
            info = self.procs.get(pid)
            if info is None:
                continue
            live_pids.add(pid)

            if self.content_resolver is not None and (do_content or not info.content.label):
                try:
                    self.content_resolver.resolve(info)
                except Exception as e:
                    # Un resolver difettoso non deve impedire di mostrare la
                    # memoria: il consumo resta visibile, l'etichetta no.
                    snap.warnings.append(f"resolver su pid {pid}: {e}")

            snap.processes.append(ProcessEntry(
                info=info,
                sample=_to_proc_sample(pm, gpu_index, now),
            ))

        self.procs.prune(live_pids)
        snap.processes.sort(key=lambda e: e.sample.dedicated_bytes, reverse=True)

        if self._balloon_bytes_provider is not None:
            try:
                snap.balloon_bytes = self._balloon_bytes_provider()
            except Exception:
                snap.balloon_bytes = 0

        return snap

    # -- riconciliazione ---------------------------------------------------
    def scale_factor(self, snap: Snapshot, gpu_index: int) -> float:
        """Rapporto fra memoria residente e somma dei committed per processo.

        Moltiplicando il `dedicated_bytes` di un processo per questo fattore si
        ottiene una **stima** della sua quota realmente residente in VRAM. E' una
        ripartizione proporzionale, non una misura: Windows non dice quali pagine
        di quale processo abbia sfrattato. La GUI deve etichettarla come stima.
        """
        gpu = next((g for g in snap.gpus if g.gpu_index == gpu_index), None)
        if gpu is None or gpu.used_bytes <= 0:
            return 1.0
        total_committed = sum(p.sample.dedicated_bytes for p in snap.processes
                              if p.sample.gpu_index == gpu_index)
        if total_committed <= 0:
            return 1.0
        return min(1.0, gpu.used_bytes / total_committed)

    def close(self) -> None:
        self.pdh.close()
        self.nvml.shutdown()


def _to_proc_sample(pm: ProcMem, gpu_index: int, ts: float) -> ProcSample:
    return ProcSample(
        pid=pm.pid,
        gpu_index=gpu_index,
        ts=ts,
        dedicated_bytes=pm.dedicated,
        shared_bytes=pm.shared,
        committed_bytes=pm.committed,
        local_bytes=pm.local,
        non_local_bytes=pm.non_local,
        engine_pct=dict(pm.engines),
    )
