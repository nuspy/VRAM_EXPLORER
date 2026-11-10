"""Rilevamento di eventi degni di autopsia.

Il servizio e' in **sola osservazione**: questo modulo non libera nulla e non
termina nulla. Riconosce, registra e notifica; ogni decisione resta all'utente.

Tre famiglie di eventi:

* **oom_suspect** — un processo che teneva VRAM sparisce mentre la GPU era sotto
  pressione. Non c'e' modo di sapere con certezza che sia morto per esaurimento
  memoria (Windows non lo dichiara), quindi il nome dice "sospetto" e
  l'incidente porta con se' i 60 s precedenti: chi teneva cosa, quanto era piena
  la GPU, cosa stava crescendo. E' l'unica ricostruzione possibile a posteriori,
  e senza registrazione continua sarebbe perduta per sempre.

* **spillover** — la memoria condivisa di un processo passa da nulla a
  significativa: sta girando su RAM di sistema invece che su VRAM, con un calo
  prestazionale che puo' arrivare a due ordini di grandezza. E' silenzioso: non
  produce errori, solo lentezza inspiegabile.

* **pressure** — occupazione oltre le soglie di guardia.

Per evitare di inondare l'utente, ogni tipo di evento ha una finestra di
raffreddamento per processo.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.models import Incident, Snapshot
from ..core.units import human_bytes, pct
from .db import HistoryStore

log = logging.getLogger("vramexp.incidents")

HISTORY_WINDOW_S = 60.0
COOLDOWN_S = 300.0


@dataclass
class _Tracked:
    pid: int
    name: str
    label: str
    gpu_index: int
    peak_dedicated: int = 0
    last_dedicated: int = 0
    last_shared: int = 0
    last_seen: float = 0.0
    spillover_notified: bool = False
    samples: deque = field(default_factory=lambda: deque(maxlen=120))


class IncidentDetector:
    """Osserva la sequenza di snapshot e produce incidenti."""

    def __init__(self, store: HistoryStore, cfg: Config | None = None,
                 on_incident=None) -> None:
        self.store = store
        self.cfg = cfg or Config()
        self.on_incident = on_incident      # callback per notifiche (tray, WS)
        self._tracked: dict[int, _Tracked] = {}
        self._recent: deque[Snapshot] = deque(maxlen=90)
        self._cooldown: dict[tuple[str, int], float] = {}
        self._pressure_state = "ok"

    # -- ciclo principale --------------------------------------------------
    def observe(self, snap: Snapshot) -> list[Incident]:
        self._recent.append(snap)
        incidents: list[Incident] = []
        now = snap.ts

        seen: set[int] = set()
        for entry in snap.processes:
            pid = entry.info.pid
            seen.add(pid)
            t = self._tracked.get(pid)
            if t is None:
                t = _Tracked(pid=pid, name=entry.info.name,
                             label=entry.info.content.label or entry.info.name,
                             gpu_index=entry.sample.gpu_index)
                self._tracked[pid] = t
            t.label = entry.info.content.label or t.label
            t.peak_dedicated = max(t.peak_dedicated, entry.sample.dedicated_bytes)
            t.last_dedicated = entry.sample.dedicated_bytes
            t.last_seen = now
            t.samples.append((now, entry.sample.dedicated_bytes, entry.sample.shared_bytes))

            inc = self._check_spillover(t, entry, snap)
            if inc:
                incidents.append(inc)

        # Processi spariti fra questo snapshot e il precedente.
        for pid, t in list(self._tracked.items()):
            if pid in seen:
                continue
            if (now - t.last_seen) < 2.0:
                continue   # attesa di un ciclo: evita falsi positivi su un campione perso
            inc = self._check_disappearance(t, snap)
            if inc:
                incidents.append(inc)
            del self._tracked[pid]

        inc = self._check_pressure(snap)
        if inc:
            incidents.append(inc)

        for i in incidents:
            try:
                i.id = self.store.add_incident(i)
            except Exception as e:
                log.warning("salvataggio incidente fallito: %s", e)
            if self.on_incident is not None:
                try:
                    self.on_incident(i)
                except Exception as e:
                    log.debug("callback incidente fallita: %s", e)
        return incidents

    # -- rilevatori --------------------------------------------------------
    def _check_disappearance(self, t: _Tracked, snap: Snapshot) -> Incident | None:
        if t.peak_dedicated < self.cfg.min_track_bytes * 4:
            return None   # processi minuscoli: la loro morte non e' interessante

        gpu = self._gpu(snap, t.gpu_index)
        if gpu is None:
            return None
        used_pct = pct(gpu.used_bytes, gpu.total_bytes)

        # Un processo che finisce normalmente sparisce anche lui: cio' che rende
        # l'evento sospetto e' che la GPU fosse quasi piena quando e' successo.
        if used_pct < self.cfg.oom_suspect_used_pct:
            kind, detail = "process_gone", (
                f"Processo terminato mentre teneva {human_bytes(t.peak_dedicated)}. "
                f"La GPU era al {used_pct:.0f}%: probabile chiusura normale.")
        else:
            kind = "oom_suspect"
            detail = (
                f"Processo scomparso con la GPU al {used_pct:.0f}% "
                f"({human_bytes(gpu.used_bytes)} su {human_bytes(gpu.total_bytes)}). "
                f"Teneva {human_bytes(t.peak_dedicated)} al picco. "
                "Possibile esaurimento della memoria: confrontare con i log "
                "dell'applicazione per confermare.")

        if not self._cooldown_ok(kind, t.pid):
            return None

        return Incident(
            ts=snap.ts, kind=kind, pid=t.pid, process_name=t.name,
            content_label=t.label, gpu_index=t.gpu_index,
            peak_dedicated_bytes=t.peak_dedicated,
            gpu_used_bytes=gpu.used_bytes, gpu_total_bytes=gpu.total_bytes,
            detail=detail, history_json=self._history_json(t.pid),
        )

    def _check_spillover(self, t: _Tracked, entry, snap: Snapshot) -> Incident | None:
        shared = entry.sample.shared_bytes
        was = t.last_shared
        t.last_shared = shared

        if shared < self.cfg.spillover_warn_bytes:
            if shared < self.cfg.spillover_warn_bytes / 2:
                t.spillover_notified = False   # rientrato: riarma la notifica
            return None
        if t.spillover_notified or was >= self.cfg.spillover_warn_bytes:
            return None
        if not self._cooldown_ok("spillover", t.pid):
            return None

        t.spillover_notified = True
        gpu = self._gpu(snap, entry.sample.gpu_index)
        return Incident(
            ts=snap.ts, kind="spillover", pid=t.pid, process_name=t.name,
            content_label=t.label, gpu_index=entry.sample.gpu_index,
            # Il picco tracciato, non il valore istantaneo: la memoria dedicata
            # oscilla mentre l'OS sposta pagine, e nel campione in cui scatta lo
            # spillover puo' risultare temporaneamente bassa.
            peak_dedicated_bytes=max(t.peak_dedicated, entry.sample.dedicated_bytes),
            gpu_used_bytes=gpu.used_bytes if gpu else 0,
            gpu_total_bytes=gpu.total_bytes if gpu else 0,
            detail=(
                f"{human_bytes(shared)} spostati sulla RAM di sistema. "
                "Il processo continua a funzionare ma accede alla memoria "
                "attraverso il bus PCIe: le prestazioni possono calare di 10-100 volte. "
                "Per farlo fallire subito invece di degradare, impostare la policy "
                "sysmem fallback su 'nessun fallback' per questa applicazione."),
            history_json=self._history_json(t.pid),
        )

    def _check_pressure(self, snap: Snapshot) -> Incident | None:
        for gpu in snap.gpus:
            if gpu.total_bytes <= 0:
                continue
            used_pct = pct(gpu.used_bytes, gpu.total_bytes)
            if used_pct >= self.cfg.critical_used_pct:
                state = "critical"
            elif used_pct >= self.cfg.warn_used_pct:
                state = "warn"
            else:
                state = "ok"

            if state == self._pressure_state or state == "ok":
                self._pressure_state = state
                continue
            self._pressure_state = state
            if not self._cooldown_ok(f"pressure_{state}", gpu.gpu_index):
                continue

            top = [e for e in snap.processes if e.sample.gpu_index == gpu.gpu_index][:3]
            who = ", ".join(f"{e.info.content.label or e.info.name} "
                            f"({human_bytes(e.sample.dedicated_bytes)})" for e in top)
            return Incident(
                ts=snap.ts, kind=f"pressure_{state}", pid=0,
                process_name="", content_label="",
                gpu_index=gpu.gpu_index,
                peak_dedicated_bytes=0,
                gpu_used_bytes=gpu.used_bytes, gpu_total_bytes=gpu.total_bytes,
                detail=f"VRAM al {used_pct:.0f}%. Principali consumatori: {who}",
                history_json="",
            )
        return None

    # -- utilita' ----------------------------------------------------------
    def _cooldown_ok(self, kind: str, pid: int) -> bool:
        key = (kind, pid)
        now = time.time()
        if (now - self._cooldown.get(key, 0.0)) < COOLDOWN_S:
            return False
        self._cooldown[key] = now
        return True

    def _gpu(self, snap: Snapshot, index: int):
        return next((g for g in snap.gpus if g.gpu_index == index), None)

    def _history_json(self, pid: int) -> str:
        """Serializza la finestra recente attorno all'evento.

        E' la sostanza dell'autopsia: senza questo, dell'incidente resterebbe
        solo la riga di riepilogo. Si include lo stato di tutti i processi, non
        solo di quello sparito, perche' la causa e' quasi sempre un altro
        processo che stava crescendo.
        """
        cutoff = time.time() - HISTORY_WINDOW_S
        frames = []
        for snap in self._recent:
            if snap.ts < cutoff:
                continue
            frames.append({
                "ts": snap.ts,
                "gpus": [{"i": g.gpu_index, "used": g.used_bytes, "total": g.total_bytes,
                          "committed": g.committed_bytes, "shared": g.shared_used_bytes}
                         for g in snap.gpus],
                "procs": [{"pid": e.info.pid, "name": e.info.name,
                           "label": e.info.content.label,
                           "ded": e.sample.dedicated_bytes,
                           "shr": e.sample.shared_bytes}
                          for e in snap.processes[:15]],
            })
        return json.dumps({"focus_pid": pid, "frames": frames}, ensure_ascii=False)
