"""Agent di introspezione, eseguito **dentro** il processo da osservare.

E' l'unico modo per vedere le singole allocazioni: dall'esterno Windows espone
solo il totale per processo. Da dentro, `torch.cuda.memory_snapshot()` elenca
ogni blocco con dimensione e stato, e con `_record_memory_history()` anche lo
stack Python che lo ha allocato — cioe' quale riga del codice ha preso quella
memoria.

Attivazione, in ordine di preferenza:

1. variabile d'ambiente `VRAMEXP_AGENT=1` prima dell'avvio del processo;
2. `import vramexp.probe.agent; vramexp.probe.agent.start()` nel proprio codice;
3. iniezione a caldo (`attach.py`), piu' invasiva.

Vincoli di progetto: nessuna dipendenza oltre a torch e pywin32, thread daemon
(non impedisce l'uscita del processo ospite), e ogni errore viene contenuto —
l'agent e' uno strumento diagnostico e non deve **mai** far cadere il programma
che sta osservando.
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Any

from . import protocol as P

_server: "_AgentServer | None" = None
_lock = threading.Lock()


class _AgentServer(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="vramexp-agent", daemon=True)
        self.pipe_name = P.pipe_name(os.getpid())
        self._stop = threading.Event()
        self._recording = False

    def run(self) -> None:
        try:
            import win32file
            import win32pipe
        except ImportError:
            return   # senza pywin32 l'agent semplicemente non parte

        while not self._stop.is_set():
            handle = None
            try:
                handle = win32pipe.CreateNamedPipe(
                    self.pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE
                    | win32pipe.PIPE_WAIT,
                    win32pipe.PIPE_UNLIMITED_INSTANCES,
                    P.MAX_MESSAGE, P.MAX_MESSAGE, 0, None)
                win32pipe.ConnectNamedPipe(handle, None)

                _, data = win32file.ReadFile(handle, P.MAX_MESSAGE)
                response = self._handle(data)
                win32file.WriteFile(handle, P.encode(response))
            except Exception:
                # Un client che chiude a meta' non deve fermare l'agent.
                pass
            finally:
                if handle is not None:
                    try:
                        win32file.CloseHandle(handle)
                    except Exception:
                        pass

    def stop(self) -> None:
        self._stop.set()

    # -- gestione dei comandi ---------------------------------------------
    def _handle(self, raw: bytes) -> dict[str, Any]:
        try:
            msg = P.decode(raw)
        except Exception as e:
            return P.err(f"richiesta non decodificabile: {e}")

        cmd = msg.get("cmd")
        params = msg.get("params") or {}
        try:
            if cmd == P.CMD_PING:
                return self._ping()
            if cmd == P.CMD_STATS:
                return self._stats(int(params.get("device", 0)))
            if cmd == P.CMD_SNAPSHOT:
                return self._snapshot(int(params.get("limit", 200)),
                                      bool(params.get("with_traces", True)))
            if cmd == P.CMD_EMPTY_CACHE:
                return self._empty_cache()
            if cmd == P.CMD_RECORD_START:
                return self._record(True, int(params.get("max_entries", 20000)))
            if cmd == P.CMD_RECORD_STOP:
                return self._record(False, 0)
            if cmd == P.CMD_MODULES:
                return self._modules(int(params.get("limit", 40)))
            return P.err(f"comando sconosciuto: {cmd}")
        except Exception as e:
            return P.err(f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])

    def _torch(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile in questo processo")
        return torch

    def _ping(self) -> dict[str, Any]:
        import sys
        info: dict[str, Any] = {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "recording": self._recording,
        }
        try:
            import torch
            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch"] = None
            info["cuda_available"] = False
        return P.ok(**info)

    def _stats(self, device: int) -> dict[str, Any]:
        torch = self._torch()
        stats = torch.cuda.memory_stats(device)
        allocated = int(stats.get("allocated_bytes.all.current", 0))
        reserved = int(stats.get("reserved_bytes.all.current", 0))
        free_driver, total = torch.cuda.mem_get_info(device)
        return P.ok(
            device=device,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            # Memoria che torch ha preso al driver ma che non sta usando: e' il
            # costo della frammentazione, spesso la ragione per cui un processo
            # occupa molto piu' di quanto sembri necessario.
            fragmentation_bytes=max(0, reserved - allocated),
            max_allocated_bytes=int(stats.get("allocated_bytes.all.peak", 0)),
            max_reserved_bytes=int(stats.get("reserved_bytes.all.peak", 0)),
            active_blocks=int(stats.get("active.all.current", 0)),
            inactive_split_blocks=int(stats.get("inactive_split.all.current", 0)),
            # Ogni retry significa che l'allocatore ha dovuto svuotare la cache e
            # riprovare: e' il segnale precoce di pressione sulla memoria.
            num_alloc_retries=int(stats.get("num_alloc_retries", 0)),
            num_ooms=int(stats.get("num_ooms", 0)),
            driver_free_bytes=int(free_driver),
            driver_total_bytes=int(total),
            alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        )

    def _snapshot(self, limit: int, with_traces: bool) -> dict[str, Any]:
        torch = self._torch()
        raw = torch.cuda.memory_snapshot()
        blocks: list[dict[str, Any]] = []
        for segment in raw:
            for block in segment.get("blocks", []):
                if block.get("state") != "active_allocated":
                    continue
                entry = {
                    "size": int(block.get("size", 0)),
                    "device": int(segment.get("device", 0)),
                    "stream": segment.get("stream"),
                    "segment_type": segment.get("segment_type"),
                    "segment_size": int(segment.get("total_size", 0)),
                }
                if with_traces:
                    entry["frames"] = _format_frames(block.get("frames") or [])
                blocks.append(entry)

        blocks.sort(key=lambda b: -b["size"])
        total = sum(b["size"] for b in blocks)
        truncated = len(blocks) > limit
        return P.ok(
            blocks=blocks[:limit],
            block_count=len(blocks),
            truncated=truncated,
            total_bytes=total,
            by_origin=_group_by_origin(blocks),
            note=("gli stack trace sono disponibili solo dopo record_start"
                  if with_traces and blocks and not blocks[0].get("frames") else ""),
        )

    def _empty_cache(self) -> dict[str, Any]:
        torch = self._torch()
        before = int(torch.cuda.memory_reserved())
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_reserved())
        return P.ok(freed_bytes=max(0, before - after),
                    reserved_before=before, reserved_after=after,
                    allocated_bytes=int(torch.cuda.memory_allocated()))

    def _record(self, enable: bool, max_entries: int) -> dict[str, Any]:
        torch = self._torch()
        if enable:
            # `_record_memory_history` e' API privata di torch, ma e' l'unica via
            # per ottenere gli stack trace delle allocazioni; il nome cambia fra
            # versioni, quindi si prova la firma nuova e si ripiega su quella vecchia.
            try:
                torch.cuda.memory._record_memory_history(
                    enabled="all", max_entries=max_entries)
            except TypeError:
                torch.cuda.memory._record_memory_history(True)
            self._recording = True
            return P.ok(recording=True, max_entries=max_entries,
                        note=("da ora in poi le nuove allocazioni porteranno lo stack "
                              "Python che le ha create; quelle gia' esistenti no"))
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except TypeError:
            torch.cuda.memory._record_memory_history(False)
        self._recording = False
        return P.ok(recording=False)

    def _modules(self, limit: int) -> dict[str, Any]:
        """Moduli torch vivi e memoria dei loro parametri.

        Rende leggibile lo snapshot: invece di "un blocco da 400 MB" si vede
        "il layer X di questo modello". Si scandisce il garbage collector perche'
        l'agent non ha alcun riferimento agli oggetti del programma ospite.
        """
        import gc
        import warnings

        import torch

        found: list[dict[str, Any]] = []
        # Scandire ogni oggetto vivo tocca anche attributi deprecati di torch, che
        # emetterebbero FutureWarning nell'output del programma ospite. L'agent
        # deve restare invisibile a chi lo ospita.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            objects = gc.get_objects()

        for obj in objects:
            try:
                if not isinstance(obj, torch.nn.Module):
                    continue
                params = sum(p.numel() * p.element_size()
                             for p in obj.parameters(recurse=True)
                             if p.is_cuda)
                if params <= 0:
                    continue
                buffers = sum(b.numel() * b.element_size()
                              for b in obj.buffers(recurse=True) if b.is_cuda)
                found.append({
                    "class": type(obj).__name__,
                    "module": type(obj).__module__,
                    "param_bytes": int(params),
                    "buffer_bytes": int(buffers),
                    "dtype": str(next(obj.parameters()).dtype) if params else None,
                    "training": bool(getattr(obj, "training", False)),
                })
            except Exception:
                continue

        # I moduli annidati contano i figli: si tengono i piu' grandi, che sono
        # tipicamente i modelli di primo livello.
        found.sort(key=lambda m: -m["param_bytes"])
        deduped: list[dict[str, Any]] = []
        for m in found:
            if any(m["param_bytes"] == d["param_bytes"] and m["class"] == d["class"]
                   for d in deduped):
                continue
            deduped.append(m)
        return P.ok(modules=deduped[:limit], module_count=len(found))


def _format_frames(frames: list[dict[str, Any]], keep: int = 6) -> list[str]:
    """Riduce lo stack alle righe di codice dell'utente.

    Uno stack di allocazione torch e' lungo decine di livelli quasi tutti interni
    alla libreria: quello che serve e' il punto del codice dell'applicazione.
    """
    out: list[str] = []
    for f in frames:
        filename = str(f.get("filename", ""))
        if any(part in filename for part in
               ("site-packages\\torch", "site-packages/torch", "<built-in>")):
            continue
        line = f.get("line")
        name = f.get("name", "?")
        out.append(f"{filename}:{line} in {name}")
        if len(out) >= keep:
            break
    if not out and frames:
        f = frames[0]
        out.append(f"{f.get('filename', '?')}:{f.get('line')} in {f.get('name', '?')}")
    return out


def _group_by_origin(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggrega le allocazioni per punto di origine nel codice."""
    groups: dict[str, dict[str, Any]] = {}
    for b in blocks:
        frames = b.get("frames") or []
        key = frames[0] if frames else "origine sconosciuta"
        g = groups.setdefault(key, {"origin": key, "bytes": 0, "count": 0})
        g["bytes"] += b["size"]
        g["count"] += 1
    return sorted(groups.values(), key=lambda g: -g["bytes"])[:25]


# -- avvio -----------------------------------------------------------------
def start() -> bool:
    """Avvia l'agent. Idempotente e silenzioso in caso di fallimento."""
    global _server
    with _lock:
        if _server is not None and _server.is_alive():
            return True
        try:
            _server = _AgentServer()
            _server.start()
            return True
        except Exception:
            _server = None
            return False


def stop() -> None:
    global _server
    with _lock:
        if _server is not None:
            _server.stop()
            _server = None


def is_running() -> bool:
    return _server is not None and _server.is_alive()


# Attivazione automatica quando la variabile d'ambiente e' impostata: consente di
# strumentare un programma senza modificarne il codice, aggiungendo l'import
# tramite `sitecustomize` oppure `python -m vramexp.probe.agent`.
if os.environ.get("VRAMEXP_AGENT", "").strip().lower() in ("1", "true", "yes", "on"):
    start()
