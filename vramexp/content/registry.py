"""Catena dei resolver: da PID a "cosa c'e' in VRAM".

I resolver sono provati in ordine di affidabilita' decrescente, dal piu'
specifico (conosce il backend e sa interrogarlo) al piu' generico (indovina dal
nome dell'eseguibile). Il primo che risponde vince, con due raffinamenti:

  * un risultato di llama.cpp viene arricchito da LM Studio quando quest'ultimo
    conosce lo stesso modello, promuovendo la confidenza a CONFIRMED;
  * l'agent di introspezione, se presente nel processo, ha sempre l'ultima
    parola: e' l'unica fonte che vede davvero dentro la memoria.

L'esito e' scritto in `info.content`. Un resolver che solleva viene isolato e non
interrompe la catena: preferiamo un'etichetta mancante a una lista vuota.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..core.config import Config
from ..core.models import Confidence, ContentInfo, ProcInfo
from .base import Resolver, close_http
from .diffusion import A1111Resolver, ComfyUIResolver, VllmResolver
from .heuristic import HeuristicResolver
from .llamacpp import LlamaCppResolver
from .lmstudio import LmStudioResolver
from .ollama import OllamaResolver

log = logging.getLogger("vramexp.content")


@dataclass(slots=True)
class _Cached:
    content: ContentInfo
    ts: float
    cmdline: str


class ContentRegistry:
    """Applica la catena di resolver e mantiene una cache per processo."""

    def __init__(self, cfg: Config | None = None, agent_lookup=None) -> None:
        self.cfg = cfg or Config()
        self.lmstudio = LmStudioResolver(self.cfg)
        # Ordine significativo: i resolver che identificano un modello preciso
        # vengono prima di quelli che identificano solo un'applicazione.
        self._chain: list[Resolver] = [
            LlamaCppResolver(),
            self.lmstudio,
            OllamaResolver(self.cfg),
            ComfyUIResolver(self.cfg),
            A1111Resolver(self.cfg),
            VllmResolver(self.cfg),
            HeuristicResolver(),
        ]
        # Iniettato da F7: restituisce i dati dell'agent per un PID, se attivo.
        self._agent_lookup = agent_lookup
        self._cache: dict[int, _Cached] = {}

    def set_agent_lookup(self, fn) -> None:
        self._agent_lookup = fn

    def resolve(self, info: ProcInfo) -> None:
        """Popola `info.content`. Non solleva mai."""
        cached = self._cache.get(info.pid)
        fresh = (cached is not None
                 and cached.cmdline == info.cmdline
                 and (time.time() - cached.ts) < self.cfg.content_refresh_s)
        if fresh and cached is not None:
            info.content = cached.content
            self._apply_agent(info)
            return

        content: ContentInfo | None = None
        for resolver in self._chain:
            try:
                if not resolver.matches(info):
                    continue
                content = resolver.resolve(info)
                if content is not None:
                    break
            except Exception as e:
                log.debug("resolver %s fallito su pid %s: %s", resolver.name, info.pid, e)
                continue

        if content is None:
            content = ContentInfo(label=info.name or f"pid {info.pid}",
                                  confidence=Confidence.UNKNOWN, source="nessuno")

        if content.backend == "llama.cpp":
            try:
                self.lmstudio.enrich(content)
            except Exception as e:
                log.debug("arricchimento LM Studio fallito: %s", e)

        info.content = content
        self._cache[info.pid] = _Cached(content=content, ts=time.time(), cmdline=info.cmdline)
        self._apply_agent(info)

    def _apply_agent(self, info: ProcInfo) -> None:
        """Sovrascrive con i dati dell'agent, che vede la memoria dall'interno."""
        if self._agent_lookup is None:
            return
        try:
            data = self._agent_lookup(info.pid)
        except Exception:
            return
        if not data:
            return
        c = info.content
        c.confidence = Confidence.CONFIRMED
        c.extra["agent"] = data
        allocated = data.get("allocated_bytes")
        reserved = data.get("reserved_bytes")
        if allocated is not None and reserved is not None:
            frag = max(0, reserved - allocated)
            c.extra["fragmentation_bytes"] = frag
            extra_detail = f"torch {_gb(allocated)} allocati / {_gb(reserved)} riservati"
            if reserved > 0 and frag / reserved > 0.25:
                extra_detail += f" · {_gb(frag)} frammentati"
            c.detail = (c.detail + " · " if c.detail else "") + extra_detail

    def forget(self, live_pids: set[int]) -> None:
        for pid in list(self._cache):
            if pid not in live_pids:
                del self._cache[pid]

    def close(self) -> None:
        close_http()


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GB"
