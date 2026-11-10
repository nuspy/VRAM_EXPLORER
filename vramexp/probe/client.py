"""Client per parlare con gli agent, dal lato del servizio.

Scopre quali processi hanno l'agent attivo (la named pipe esiste) e ne interroga
lo stato. La scoperta e' economica — tentare di aprire una pipe inesistente
fallisce immediatamente — quindi si puo' ripetere a intervalli regolari senza
costo apprezzabile.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import protocol as P

log = logging.getLogger("vramexp.probe")

DISCOVERY_TTL = 5.0
CALL_TIMEOUT_MS = 4000


class AgentClient:
    def __init__(self) -> None:
        self._known: dict[int, float] = {}     # pid -> ultima conferma
        self._cache: dict[int, dict[str, Any]] = {}
        self._cache_ts: dict[int, float] = {}
        self._lock = threading.Lock()

    # -- scoperta ----------------------------------------------------------
    def has_agent(self, pid: int) -> bool:
        with self._lock:
            seen = self._known.get(pid, 0.0)
        if (time.time() - seen) < DISCOVERY_TTL:
            return True
        resp = self.call(pid, P.CMD_PING, quiet=True)
        if resp and resp.get("ok"):
            with self._lock:
                self._known[pid] = time.time()
            return True
        with self._lock:
            self._known.pop(pid, None)
        return False

    def discover(self, pids: list[int]) -> dict[int, dict[str, Any]]:
        """Interroga un gruppo di PID e restituisce quelli con agent attivo."""
        found: dict[int, dict[str, Any]] = {}
        for pid in pids:
            resp = self.call(pid, P.CMD_PING, quiet=True)
            if resp and resp.get("ok"):
                found[pid] = resp
                with self._lock:
                    self._known[pid] = time.time()
        return found

    # -- interrogazione ----------------------------------------------------
    def summary(self, pid: int, ttl: float = 3.0) -> dict[str, Any] | None:
        """Riepilogo per la GUI: allocato, riservato, frammentazione.

        Con cache breve perche' viene chiesto per ogni processo a ogni refresh.
        """
        now = time.time()
        with self._lock:
            if pid in self._cache and (now - self._cache_ts.get(pid, 0)) < ttl:
                return self._cache[pid]
        resp = self.call(pid, P.CMD_STATS, quiet=True)
        if not resp or not resp.get("ok"):
            return None
        data = {k: v for k, v in resp.items() if k != "ok"}
        with self._lock:
            self._cache[pid] = data
            self._cache_ts[pid] = now
        return data

    def call(self, pid: int, command: str, quiet: bool = False,
             **params: Any) -> dict[str, Any] | None:
        """Invia un comando all'agent. `None` se l'agent non risponde."""
        try:
            import win32file
            import win32pipe
        except ImportError:
            return None

        name = P.pipe_name(pid)
        handle = None
        try:
            handle = win32file.CreateFile(
                name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0, None, win32file.OPEN_EXISTING, 0, None)
            win32pipe.SetNamedPipeHandleState(
                handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            win32file.WriteFile(handle, P.encode(P.request(command, **params)))
            _, data = win32file.ReadFile(handle, P.MAX_MESSAGE)
            return P.decode(data)
        except Exception as e:
            if not quiet:
                log.debug("chiamata all'agent del pid %s fallita: %s", pid, e)
            return None
        finally:
            if handle is not None:
                try:
                    win32file.CloseHandle(handle)
                except Exception:
                    pass

    def forget(self, live_pids: set[int]) -> None:
        with self._lock:
            for pid in list(self._known):
                if pid not in live_pids:
                    self._known.pop(pid, None)
                    self._cache.pop(pid, None)
                    self._cache_ts.pop(pid, None)
