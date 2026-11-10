"""Collegamento della GUI al servizio.

Due modalita', decise all'avvio e visibili all'utente nella barra di stato:

* **collegata** — il servizio e' in esecuzione; la GUI riceve i campioni via
  WebSocket e ha accesso a tutto lo storico;
* **autonoma** — nessun servizio: la GUI campiona da sola. Funziona tutto tranne
  lo storico precedente all'avvio, perche' nessuno lo stava registrando.

La distinzione va mostrata, non nascosta: un grafico storico vuoto ha
spiegazioni molto diverse nei due casi.

Il campionamento avviene sempre fuori dal thread dell'interfaccia. Leggere i
contatori PDH richiede alcuni millisecondi e farlo nel thread della GUI
produrrebbe micro-scatti a ogni aggiornamento.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import httpx
from PySide6.QtCore import QObject, QThread, Signal

from vramexp.core.config import Config, get_or_create_token

log = logging.getLogger("vramexp.gui.client")


class ServiceClient(QObject):
    """Sorgente dati unificata, indipendente dalla modalita'."""

    snapshot_ready = Signal(dict)
    incident_received = Signal(dict)
    connection_changed = Signal(bool, str)

    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Config.load()
        self.base = f"http://{self.cfg.api_host}:{self.cfg.api_port}"
        self.token = get_or_create_token() if self.cfg.api_require_token else None
        self.connected = False
        self._local_service = None
        self._worker: _SamplerWorker | None = None
        self._thread: QThread | None = None
        self._http = httpx.Client(timeout=20.0,
                                  headers={"x-vramexp-token": self.token or ""})

    # -- ciclo di vita -----------------------------------------------------
    def start(self) -> None:
        self.connected = self._probe()
        if self.connected:
            self.connection_changed.emit(True, "collegata al servizio")
        else:
            self.connection_changed.emit(
                False, "servizio non attivo — modalità autonoma, senza storico pregresso")

        self._thread = QThread()
        self._worker = _SamplerWorker(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.snapshot.connect(self.snapshot_ready)
        self._worker.incident.connect(self.incident_received)
        self._worker.connection.connect(self._on_connection)
        self._thread.start()

    def _on_connection(self, ok: bool, message: str) -> None:
        if ok != self.connected:
            self.connected = ok
            self.connection_changed.emit(ok, message)

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        if self._local_service is not None:
            self._local_service.stop()
        self._http.close()

    def _probe(self) -> bool:
        try:
            return self._http.get(f"{self.base}/health", timeout=1.5).status_code == 200
        except Exception:
            return False

    @property
    def service(self):
        """Istanza locale, creata solo se il servizio non c'e'."""
        if self._local_service is None:
            from vramexp.service import VramService
            self._local_service = VramService(self.cfg)
        return self._local_service

    # -- richieste ---------------------------------------------------------
    def get(self, path: str, **params: Any) -> Any:
        if self.connected:
            r = self._http.get(f"{self.base}{path}", params=params)
            r.raise_for_status()
            return r.json()
        return self._local_get(path, params)

    def post(self, path: str, payload: Any = None) -> Any:
        if self.connected:
            r = self._http.post(f"{self.base}{path}", json=payload)
            return r.json()
        return self._local_post(path, payload or {})

    def _local_get(self, path: str, params: dict) -> Any:
        """Instrada le richieste all'istanza locale quando il servizio manca.

        Copre solo gli endpoint che la GUI usa davvero: aggiungerne di nuovi
        senza mapparli qui produce un errore esplicito, non un silenzio.
        """
        s = self.service
        if path == "/status":
            return s.status()
        if path == "/processes":
            return {"processes": s.processes(params.get("include_system", True))}
        if path.startswith("/process/") and path.endswith("/allocations"):
            pid = int(path.split("/")[2])
            return s.agent_client.call(pid, "snapshot", limit=params.get("limit", 100)) or {}
        if path.startswith("/process/") and path.endswith("/modules"):
            pid = int(path.split("/")[2])
            return s.agent_client.call(pid, "modules") or {}
        if path.startswith("/process/"):
            pid = int(path.rsplit("/", 1)[1])
            entry = s.find_process(pid)
            if entry is None:
                return {}
            d = entry.to_dict()
            d["actions"] = [o.to_dict() for o in s.unloader.options_for(entry.info)]
            d["agent"] = s.agent_client.summary(pid)
            return d
        if path == "/history/gpu":
            since = time.time() - float(params.get("minutes", 60)) * 60
            return {"samples": s.store.gpu_history(since, gpu_index=params.get("gpu", 0))}
        if path == "/history/processes":
            since = time.time() - float(params.get("minutes", 60)) * 60
            rows = s.store.proc_history(since, top_n=params.get("top", 12))
            pids = {int(r["pid"]) for r in rows}
            return {"samples": rows, "processes": s.store.process_names(pids)}
        if path == "/incidents":
            return {"incidents": s.store.incidents(limit=params.get("limit", 50))}
        if path.startswith("/incidents/"):
            return s.store.incident(int(path.rsplit("/", 1)[1])) or {}
        if path == "/library":
            index = s.library_index(rescan=params.get("rescan", False))
            q = params.get("query")
            models = index.find(q) if q else index.models
            return {"count": len(models), "total_bytes": index.total_bytes(),
                    "models": [m.to_dict() for m in models[:params.get("limit", 100)]],
                    "errors": index.errors}
        if path == "/plan":
            return s.plan(params.get("model", ""), params.get("context"),
                          kv_type=params.get("kv_type", "f16"))
        if path == "/profiles":
            return {"profiles": [p.to_dict() for p in s.profiles.list()]}
        if path == "/policy":
            return s.policy.get(params.get("exe")).to_dict()
        if path == "/reserve":
            return s.balloon.state()
        if path.startswith("/probe/eligibility/"):
            from vramexp.probe.attach import check_eligibility
            pid = int(path.rsplit("/", 1)[1])
            eligible, reason = check_eligibility(pid)
            return {"pid": pid, "eligible": eligible, "reason": reason,
                    "hot_attach_enabled": self.cfg.enable_hot_attach}
        if path == "/config":
            return s.cfg.to_dict()
        if path == "/stats":
            return s.store.stats()
        raise NotImplementedError(f"endpoint {path} non gestito in modalità autonoma")

    def _local_post(self, path: str, payload: dict) -> Any:
        s = self.service
        if path == "/action":
            entry = s.find_process(int(payload["pid"]))
            if entry is None:
                return {"ok": False, "message": "processo non trovato"}
            return s.unloader.execute(entry.info, payload["kind"],
                                      payload.get("params"),
                                      payload.get("confirmed", False)).to_dict()
        if path == "/reserve":
            return s.balloon.set_reserve(int(payload["size"]))
        if path == "/policy":
            ok, message = s.policy.set(int(payload["value"]), payload.get("exe"))
            return {"ok": ok, "message": message}
        if path == "/profiles/preview":
            return s.profiles.preview(payload["name"], s.snapshot())
        if path == "/profiles/apply":
            return s.profiles.apply(payload["name"], s.snapshot(),
                                    payload.get("confirmed", False))
        if path == "/probe/attach":
            from vramexp.probe.attach import attach
            return attach(int(payload["pid"]), payload.get("confirmed", False),
                          self.cfg.enable_hot_attach).to_dict()
        if path.startswith("/probe/record/"):
            pid = int(path.rsplit("/", 1)[1])
            return s.agent_client.call(pid, "record_start") or {"ok": False}
        if path == "/config":
            for k, v in payload.items():
                if hasattr(s.cfg, k):
                    setattr(s.cfg, k, v)
            s.cfg.save()
            return {"ok": True}
        raise NotImplementedError(f"endpoint {path} non gestito in modalità autonoma")


class _SamplerWorker(QObject):
    """Produce campioni: dal WebSocket del servizio o campionando in proprio."""

    snapshot = Signal(dict)
    incident = Signal(dict)
    connection = Signal(bool, str)

    def __init__(self, client: ServiceClient) -> None:
        super().__init__()
        self.client = client
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            if self.client._probe():
                self.connection.emit(True, "collegata al servizio")
                self._stream()
            else:
                self.connection.emit(
                    False, "servizio non attivo — modalità autonoma")
                self._sample_locally()

    def _stream(self) -> None:
        """Riceve i campioni dal WebSocket del servizio."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._poll_http()
            return

        url = f"ws://{self.client.cfg.api_host}:{self.client.cfg.api_port}/live"
        if self.client.token:
            url += f"?token={self.client.token}"
        try:
            with connect(url, open_timeout=5) as ws:
                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=5)
                    except TimeoutError:
                        continue
                    message = json.loads(raw)
                    if message.get("type") == "snapshot":
                        self.snapshot.emit(message["data"])
                    elif message.get("type") == "incident":
                        self.incident.emit(message["data"])
        except Exception as e:
            log.debug("websocket interrotto: %s", e)
            # Non si passa subito alla modalita' autonoma: il servizio potrebbe
            # essersi solo riavviato, e crearne una seconda istanza
            # raddoppierebbe scritture e campionamento sullo stesso database.
            self._stop.wait(2.0)

    def _poll_http(self) -> None:
        while not self._stop.is_set():
            try:
                self.snapshot.emit(self.client.get("/snapshot"))
            except Exception:
                return
            self._stop.wait(self.client.cfg.sample_interval_s)

    def _sample_locally(self) -> None:
        service = self.client.service
        while not self._stop.is_set():
            if self.client._probe():
                return   # il servizio e' comparso: si passa allo streaming
            try:
                snap = service.sampler.sample()
                service.store.write_snapshot(snap)
                for inc in service.detector.observe(snap):
                    self.incident.emit(inc.to_dict())
                self.snapshot.emit(snap.to_dict())
            except Exception as e:
                log.warning("campionamento locale fallito: %s", e)
            self._stop.wait(self.client.cfg.sample_interval_s)
