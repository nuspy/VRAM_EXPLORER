"""Motore del servizio: campiona, registra, rileva, e serve i client.

Tiene insieme tutti i sottosistemi ed espone un'unica facciata usata sia
dall'API sia dalla CLI in modalita' diretta. La regola che governa il
comportamento e' una sola: **il ciclo di campionamento non deve mai fermarsi**.
Qualunque componente puo' fallire — un backend che non risponde, l'agent di un
processo morto, la manutenzione dello storico — senza interrompere la raccolta,
perche' la continuita' dello storico e' l'unica cosa che non si puo' recuperare
a posteriori.

Il servizio e' in **sola osservazione**: rileva e notifica, non agisce mai da
solo. Le azioni esistono ma partono esclusivamente da una richiesta esplicita.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from .collect.sampler import Sampler
from .content.registry import ContentRegistry
from .control.balloon import free_vram_bytes
from .control.fallback_policy import FallbackPolicyManager
from .control.unloader import Unloader
from .core.config import Config
from .core.models import Incident, Snapshot
from .core.units import human_bytes, pct
from .planner.library_scan import LibraryScanner
from .planner.profiles import ProfileManager
from .planner.whatif import Planner
from .probe.client import AgentClient
from .store.db import HistoryStore
from .store.incidents import IncidentDetector
from .store.retention import RetentionManager

log = logging.getLogger("vramexp.service")


class BalloonController:
    """Gestisce il processo figlio che detiene la riserva di VRAM.

    La riserva vive in un processo separato perche' il contesto CUDA appartiene a
    chi lo crea: tenerla nel servizio significherebbe caricare il runtime CUDA
    dentro il processo che deve restare leggero e sempre vivo, e perderla a ogni
    riavvio del servizio.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._proc = None
        self._target_bytes = 0
        self._last_state: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._target_bytes
            return 0

    def set_reserve(self, target_bytes: int) -> dict[str, Any]:
        import json
        import subprocess
        import sys
        from pathlib import Path

        with self._lock:
            self._stop_locked()
            self._target_bytes = max(0, int(target_bytes))
            if self._target_bytes == 0:
                self._last_state = {"allocated_bytes": 0, "requested_bytes": 0,
                                    "message": "riserva rilasciata"}
                return dict(self._last_state)

            script = Path(__file__).resolve().parent.parent / "balloon_main.py"
            argv = [sys.executable, str(script),
                    "--bytes", str(self._target_bytes),
                    "--chunk", str(self.cfg.balloon_chunk_bytes)]
            try:
                self._proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, creationflags=_no_window())
            except OSError as e:
                self._last_state = {"allocated_bytes": 0, "error": str(e)}
                return dict(self._last_state)

            # Il figlio stampa il proprio stato appena la riserva e' in piedi:
            # attenderlo consente di riportare l'esito reale invece di un
            # ottimistico "richiesto".
            state: dict[str, Any] = {}
            deadline = time.time() + 30.0
            while time.time() < deadline:
                line = self._proc.stdout.readline() if self._proc.stdout else ""
                if not line:
                    if self._proc.poll() is not None:
                        break
                    continue
                line = line.strip()
                if line.startswith("{"):
                    try:
                        state = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if not state:
                state = {"allocated_bytes": 0,
                         "error": "il processo di riserva non ha risposto"}
                self._stop_locked()
            self._last_state = state
            return dict(state)

    def state(self) -> dict[str, Any]:
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
            out = dict(self._last_state)
            out["active"] = alive
            if not alive:
                out["allocated_bytes"] = 0
            return out

    def _stop_locked(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
            self._target_bytes = 0


class VramService:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.store = HistoryStore()
        self.agent_client = AgentClient()
        self.content = ContentRegistry(self.cfg, agent_lookup=self._agent_lookup)
        self.sampler = Sampler(self.cfg, content_resolver=self.content)
        self.balloon = BalloonController(self.cfg)
        self.unloader = Unloader(self.cfg, agent_client=self.agent_client,
                                 free_bytes_fn=lambda: _free_bytes(self.sampler))
        self.policy = FallbackPolicyManager()
        self.retention = RetentionManager(self.store, self.cfg)
        self.detector = IncidentDetector(self.store, self.cfg,
                                         on_incident=self._on_incident)
        self.library = LibraryScanner(self.cfg.model_dirs)
        self.planner = Planner(store=self.store, unloader=self.unloader)
        self.profiles = ProfileManager(unloader=self.unloader,
                                       balloon_ctl=self.balloon,
                                       policy=self.policy,
                                       agent_client=self.agent_client)

        self.sampler.set_balloon_provider(lambda: self.balloon.reserved_bytes)

        self._latest: Snapshot | None = None
        self._latest_lock = threading.Lock()
        self._subscribers: list[Callable[[Snapshot], None]] = []
        self._incident_subs: list[Callable[[Incident], None]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._library_index = None
        self.started_at = time.time()
        self.sample_count = 0
        self.last_error: str | None = None

    # -- ciclo di campionamento -------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vramexp-sampler",
                                        daemon=True)
        self._thread.start()
        log.info("campionamento avviato (intervallo %.1f s)", self.cfg.sample_interval_s)

    def _loop(self) -> None:
        consecutive_errors = 0
        while not self._stop.is_set():
            started = time.time()
            try:
                snap = self.sampler.sample()
                self._enrich_with_agents(snap)

                with self._latest_lock:
                    self._latest = snap
                self.sample_count += 1
                consecutive_errors = 0
                self.last_error = None

                self.store.write_snapshot(snap)
                self.detector.observe(snap)
                self.retention.maybe_run()
                self._notify(snap)
            except Exception as e:
                consecutive_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("errore nel ciclo di campionamento (%d consecutivi)",
                              consecutive_errors)
                # Dopo una serie di errori si rallenta invece di insistere a
                # piena frequenza: se la causa e' sistemica, martellare peggiora.
                if consecutive_errors > 5:
                    self._stop.wait(min(30.0, consecutive_errors * 2.0))

            elapsed = time.time() - started
            self._stop.wait(max(0.05, self.cfg.sample_interval_s - elapsed))

    def _enrich_with_agents(self, snap: Snapshot) -> None:
        """Aggiunge i dati dell'agent ai processi che lo espongono."""
        live = {e.info.pid for e in snap.processes}
        for entry in snap.processes:
            data = self.agent_client.summary(entry.info.pid)
            if data:
                entry.info.content.extra["agent"] = data
        self.agent_client.forget(live)

    def _agent_lookup(self, pid: int) -> dict[str, Any] | None:
        return self.agent_client.summary(pid)

    def _notify(self, snap: Snapshot) -> None:
        for fn in list(self._subscribers):
            try:
                fn(snap)
            except Exception as e:
                log.debug("subscriber fallito: %s", e)

    def _on_incident(self, inc: Incident) -> None:
        log.warning("[%s] %s — %s", inc.kind, inc.process_name or "sistema", inc.detail)
        for fn in list(self._incident_subs):
            try:
                fn(inc)
            except Exception as e:
                log.debug("subscriber incidenti fallito: %s", e)

    def subscribe(self, fn: Callable[[Snapshot], None]) -> Callable[[], None]:
        self._subscribers.append(fn)
        return lambda: self._subscribers.remove(fn) if fn in self._subscribers else None

    def subscribe_incidents(self, fn: Callable[[Incident], None]) -> Callable[[], None]:
        self._incident_subs.append(fn)
        return lambda: (self._incident_subs.remove(fn)
                        if fn in self._incident_subs else None)

    # -- accesso ai dati ---------------------------------------------------
    def snapshot(self, wait: float = 3.0) -> Snapshot:
        """Ultimo snapshot. Se il ciclo non e' partito, ne produce uno al volo."""
        with self._latest_lock:
            if self._latest is not None:
                return self._latest
        if self._thread is None or not self._thread.is_alive():
            snap = self.sampler.sample()
            with self._latest_lock:
                self._latest = snap
            return snap
        deadline = time.time() + wait
        while time.time() < deadline:
            time.sleep(0.1)
            with self._latest_lock:
                if self._latest is not None:
                    return self._latest
        return Snapshot(warnings=["nessun campione disponibile"])

    def find_process(self, pid: int):
        snap = self.snapshot()
        return next((e for e in snap.processes if e.info.pid == pid), None)

    def status(self) -> dict[str, Any]:
        snap = self.snapshot()
        gpus = []
        for g in snap.gpus:
            info = next((i for i in snap.gpu_info if i.index == g.gpu_index), None)
            scale = self.sampler.scale_factor(snap, g.gpu_index)
            gpus.append({
                **g.to_dict(),
                "name": info.name if info else f"GPU {g.gpu_index}",
                "used_pct": pct(g.used_bytes, g.total_bytes),
                "residency_scale": scale,
                "residency_note": (
                    "i valori per processo sono memoria *committed*; moltiplicati per "
                    f"{scale:.2f} danno una stima della quota residente in VRAM"),
            })
        return {
            "ts": snap.ts,
            "gpus": gpus,
            "process_count": len(snap.processes),
            "balloon": self.balloon.state(),
            "warnings": snap.warnings,
            "health": {
                **self.sampler.health(),
                "uptime_s": time.time() - self.started_at,
                "samples": self.sample_count,
                "last_error": self.last_error,
                "nvapi_available": self.policy.available,
                "observation_only": True,
            },
        }

    def processes(self, include_system: bool = True, gpu_index: int | None = None
                  ) -> list[dict[str, Any]]:
        from .collect.proc_source import is_system_process

        snap = self.snapshot()
        out = []
        for entry in snap.processes:
            if gpu_index is not None and entry.sample.gpu_index != gpu_index:
                continue
            if not include_system and is_system_process(entry.info):
                continue
            d = entry.to_dict()
            d["actions"] = [o.to_dict() for o in self.unloader.options_for(entry.info)]
            d["has_agent"] = self.agent_client.has_agent(entry.info.pid)
            out.append(d)
        return out

    def library_index(self, rescan: bool = False):
        if self._library_index is None or rescan:
            self._library_index = self.library.scan()
        return self._library_index

    def plan(self, model_query: str, context: int | None = None,
             gpu_index: int = 0, gpu_layers: int | None = None,
             kv_type: str = "f16") -> dict[str, Any]:
        index = self.library_index()
        matches = index.find(model_query)
        if not matches:
            return {"ok": False,
                    "error": f"nessun modello corrisponde a «{model_query}»",
                    "hint": "usare /library per elencare i modelli indicizzati"}
        model = matches[0]
        result = self.planner.plan(model, self.snapshot(), context=context,
                                   gpu_index=gpu_index, gpu_layers=gpu_layers,
                                   kv_type=kv_type)
        return {"ok": True, "model": model.to_dict(), **result.to_dict(),
                "other_matches": [m.name for m in matches[1:6]]}

    def record_model_load(self, entry) -> None:
        """Registra la memoria realmente occupata da un modello identificato.

        Alimenta il planner con misure invece che stime; viene chiamata quando un
        processo con contenuto riconosciuto raggiunge uno stato stabile.
        """
        c = entry.info.content
        if not c.model_name or not c.file_size_bytes:
            return
        key = f"{c.model_name}|{c.file_size_bytes}"
        try:
            self.store.record_model_observation(
                key, entry.sample.dedicated_bytes, c.context_length,
                c.quantization, c.backend)
        except Exception as e:
            log.debug("registrazione osservazione fallita: %s", e)

    # -- arresto -----------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.balloon.stop()
        self.sampler.close()
        self.content.close()
        self.policy.close()
        self.store.close()
        log.info("servizio arrestato")


def _free_bytes(sampler: Sampler) -> int:
    samples = sampler.nvml.sample()
    return samples[0].free_bytes if samples else free_vram_bytes()[0]


def _no_window() -> int:
    import sys
    return 0x08000000 if sys.platform == "win32" else 0
