"""API REST e WebSocket del servizio, su loopback.

Rende il servizio pilotabile da script, dalla CLI, dalla GUI e da altri progetti
sulla stessa macchina. Il binding e' **esclusivamente** su 127.0.0.1 e ogni
richiesta richiede un token generato al primo avvio e conservato nel profilo
utente: la superficie e' locale, ma le operazioni che espone (terminare
processi, cambiare impostazioni del driver) meritano comunque una barriera.

Le operazioni distruttive richiedono `confirmed: true` esplicito nel corpo della
richiesta, esattamente come dalla GUI. Un client che dimentica il flag riceve un
rifiuto, non un processo terminato.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from ..core.config import Config, get_or_create_token
from ..core.units import parse_size
from ..service import VramService

log = logging.getLogger("vramexp.api")


# --- corpi delle richieste -------------------------------------------------
class ActionRequest(BaseModel):
    pid: int
    kind: str = Field(description="backend_api | agent_empty_cache | trim | suspend | kill")
    confirmed: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ReserveRequest(BaseModel):
    size: str | int = Field(description='byte oppure stringa tipo "8G"')


class PolicyRequest(BaseModel):
    value: int = Field(description="0 = consenti RAM di sistema, 1 = fai fallire")
    exe: str | None = Field(default=None,
                            description="eseguibile per un profilo specifico")


class ProfileRequest(BaseModel):
    name: str
    confirmed: bool = False


class AttachRequest(BaseModel):
    pid: int
    confirmed: bool = False


def create_app(service: VramService, cfg: Config | None = None) -> FastAPI:
    cfg = cfg or service.cfg
    token = get_or_create_token() if cfg.api_require_token else None

    app = FastAPI(title="VRAM Explorer", version="1.0.0",
                  description="Esplorazione e gestione della VRAM (servizio locale)")

    async def auth(request: Request) -> None:
        if not cfg.api_require_token:
            return
        supplied = (request.headers.get("x-vramexp-token")
                    or request.query_params.get("token")
                    or (request.headers.get("authorization") or "").removeprefix("Bearer ").strip())
        if supplied != token:
            raise HTTPException(status_code=401, detail="token mancante o non valido")

    guard = [Depends(auth)]

    # --- stato -------------------------------------------------------------
    @app.get("/status", dependencies=guard)
    def status() -> dict:
        return service.status()

    @app.get("/snapshot", dependencies=guard)
    def snapshot() -> dict:
        return service.snapshot().to_dict()

    @app.get("/processes", dependencies=guard)
    def processes(include_system: bool = True, gpu: int | None = None) -> dict:
        return {"processes": service.processes(include_system, gpu)}

    @app.get("/process/{pid}", dependencies=guard)
    def process(pid: int) -> dict:
        entry = service.find_process(pid)
        if entry is None:
            raise HTTPException(404, f"nessun processo {pid} sta usando la GPU")
        d = entry.to_dict()
        d["actions"] = [o.to_dict() for o in service.unloader.options_for(entry.info)]
        d["agent"] = service.agent_client.summary(pid)
        return d

    @app.get("/process/{pid}/allocations", dependencies=guard)
    def allocations(pid: int, limit: int = 100, with_traces: bool = True) -> dict:
        """Singole allocazioni, disponibili solo con l'agent attivo nel processo."""
        if not service.agent_client.has_agent(pid):
            raise HTTPException(404, (
                "agent non attivo in questo processo. Avviarlo con VRAMEXP_AGENT=1 "
                "oppure usare /probe/attach se l'iniezione a caldo e' abilitata."))
        resp = service.agent_client.call(pid, "snapshot", limit=limit,
                                         with_traces=with_traces)
        if not resp or not resp.get("ok"):
            raise HTTPException(502, str((resp or {}).get("error", "agent non risponde")))
        return resp

    @app.get("/process/{pid}/modules", dependencies=guard)
    def modules(pid: int) -> dict:
        resp = service.agent_client.call(pid, "modules")
        if not resp or not resp.get("ok"):
            raise HTTPException(404, "agent non disponibile in questo processo")
        return resp

    # --- storico -----------------------------------------------------------
    @app.get("/history/gpu", dependencies=guard)
    def history_gpu(minutes: float = Query(60, ge=0.1, le=44640), gpu: int = 0) -> dict:
        since = time.time() - minutes * 60
        return {"samples": service.store.gpu_history(since, gpu_index=gpu)}

    @app.get("/history/processes", dependencies=guard)
    def history_processes(minutes: float = Query(60, ge=0.1, le=44640),
                          pid: int | None = None, top: int = 12) -> dict:
        since = time.time() - minutes * 60
        rows = service.store.proc_history(since, pid=pid, top_n=top)
        pids = {int(r["pid"]) for r in rows}
        return {"samples": rows, "processes": service.store.process_names(pids)}

    @app.get("/incidents", dependencies=guard)
    def incidents(limit: int = 50, hours: float | None = None) -> dict:
        since = (time.time() - hours * 3600) if hours else None
        return {"incidents": service.store.incidents(limit=limit, since=since)}

    @app.get("/incidents/{incident_id}", dependencies=guard)
    def incident(incident_id: int) -> dict:
        row = service.store.incident(incident_id)
        if row is None:
            raise HTTPException(404, "incidente inesistente")
        return row

    @app.get("/stats", dependencies=guard)
    def stats() -> dict:
        return service.store.stats()

    # --- azioni ------------------------------------------------------------
    @app.post("/action", dependencies=guard)
    def action(req: ActionRequest) -> dict:
        entry = service.find_process(req.pid)
        if entry is None:
            raise HTTPException(404, f"nessun processo {req.pid} sta usando la GPU")
        result = service.unloader.execute(entry.info, req.kind, req.params, req.confirmed)
        code = 200 if result.ok else 409
        return JSONResponse(result.to_dict(), status_code=code)

    @app.post("/reserve", dependencies=guard)
    def reserve(req: ReserveRequest) -> dict:
        size = parse_size(req.size) if isinstance(req.size, str) else int(req.size)
        return service.balloon.set_reserve(size)

    @app.get("/reserve", dependencies=guard)
    def reserve_state() -> dict:
        return service.balloon.state()

    @app.get("/policy", dependencies=guard)
    def policy(exe: str | None = None) -> dict:
        return service.policy.get(exe).to_dict()

    @app.post("/policy", dependencies=guard)
    def set_policy(req: PolicyRequest) -> dict:
        ok, message = service.policy.set(req.value, req.exe)
        return JSONResponse({"ok": ok, "message": message},
                            status_code=200 if ok else 409)

    # --- pianificazione ----------------------------------------------------
    @app.get("/library", dependencies=guard)
    def library(rescan: bool = False, query: str | None = None, limit: int = 100) -> dict:
        index = service.library_index(rescan=rescan)
        models = index.find(query) if query else index.models
        return {"count": len(models), "total_bytes": index.total_bytes(),
                "scanned_at": index.scanned_at, "errors": index.errors,
                "models": [m.to_dict() for m in models[:limit]]}

    @app.get("/plan", dependencies=guard)
    def plan(model: str, context: int | None = None, gpu: int = 0,
             gpu_layers: int | None = None, kv_type: str = "f16") -> dict:
        result = service.plan(model, context, gpu, gpu_layers, kv_type)
        return JSONResponse(result, status_code=200 if result.get("ok") else 404)

    @app.get("/profiles", dependencies=guard)
    def profiles() -> dict:
        return {"profiles": [p.to_dict() for p in service.profiles.list()]}

    @app.post("/profiles/preview", dependencies=guard)
    def profile_preview(req: ProfileRequest) -> dict:
        result = service.profiles.preview(req.name, service.snapshot())
        return JSONResponse(result, status_code=200 if result.get("ok") else 404)

    @app.post("/profiles/apply", dependencies=guard)
    def profile_apply(req: ProfileRequest) -> dict:
        result = service.profiles.apply(req.name, service.snapshot(), req.confirmed)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    # --- probe -------------------------------------------------------------
    @app.get("/probe/eligibility/{pid}", dependencies=guard)
    def probe_eligibility(pid: int) -> dict:
        from ..probe.attach import check_eligibility
        eligible, reason = check_eligibility(pid)
        return {"pid": pid, "eligible": eligible, "reason": reason,
                "hot_attach_enabled": service.cfg.enable_hot_attach,
                "safe_alternative": "avviare il processo con VRAMEXP_AGENT=1"}

    @app.post("/probe/attach", dependencies=guard)
    def probe_attach(req: AttachRequest) -> dict:
        from ..probe.attach import attach
        result = attach(req.pid, confirmed=req.confirmed,
                        enabled=service.cfg.enable_hot_attach)
        return JSONResponse(result.to_dict(), status_code=200 if result.ok else 409)

    @app.post("/probe/record/{pid}", dependencies=guard)
    def probe_record(pid: int, enable: bool = True) -> dict:
        cmd = "record_start" if enable else "record_stop"
        resp = service.agent_client.call(pid, cmd)
        if not resp or not resp.get("ok"):
            raise HTTPException(404, "agent non disponibile in questo processo")
        return resp

    # --- configurazione ----------------------------------------------------
    @app.get("/config", dependencies=guard)
    def get_config() -> dict:
        return service.cfg.to_dict()

    @app.post("/config", dependencies=guard)
    def set_config(values: dict[str, Any]) -> dict:
        known = set(service.cfg.to_dict())
        applied = {}
        for k, v in values.items():
            if k in known:
                setattr(service.cfg, k, v)
                applied[k] = v
        service.cfg.save()
        return {"ok": True, "applied": applied,
                "ignored": [k for k in values if k not in known]}

    # --- flusso in tempo reale --------------------------------------------
    @app.websocket("/live")
    async def live(ws: WebSocket) -> None:
        supplied = ws.query_params.get("token")
        if cfg.api_require_token and supplied != token:
            await ws.close(code=4401)
            return
        await ws.accept()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        def on_snapshot(snap) -> None:
            # Il campionamento avviene in un thread: si passa al loop asincrono
            # in modo thread-safe, scartando i frame se il client e' lento invece
            # di accumularli e far crescere la memoria.
            try:
                loop.call_soon_threadsafe(_offer, queue, {"type": "snapshot",
                                                          "data": snap.to_dict()})
            except RuntimeError:
                pass

        def on_incident(inc) -> None:
            try:
                loop.call_soon_threadsafe(_offer, queue, {"type": "incident",
                                                          "data": inc.to_dict()})
            except RuntimeError:
                pass

        unsub = service.subscribe(on_snapshot)
        unsub_inc = service.subscribe_incidents(on_incident)
        try:
            while True:
                message = await queue.get()
                await ws.send_json(message)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.debug("websocket chiuso: %s", e)
        finally:
            unsub()
            unsub_inc()

    @app.get("/health")
    def health() -> dict:
        """Senza autenticazione: serve al supervisore per sapere se il processo
        e' vivo, e non espone alcun dato sensibile."""
        return {"ok": True, "uptime_s": time.time() - service.started_at,
                "samples": service.sample_count}

    return app


def _offer(queue: asyncio.Queue, item: dict) -> None:
    if queue.full():
        try:
            queue.get_nowait()   # scarta il piu' vecchio: conta l'attualita'
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass
