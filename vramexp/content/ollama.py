"""Ollama: modelli residenti via `/api/ps`.

`/api/ps` e' particolarmente prezioso perche' e' l'unica API fra quelle
supportate che dichiara **quanta** VRAM occupa ciascun modello (`size_vram`) e
quando scadra' (`expires_at`). Permette quindi di attribuire la memoria di un
processo `ollama.exe` ai singoli modelli che ospita, cosa che per gli altri
backend non e' possibile.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from ..core.config import Config
from ..core.models import Confidence, ContentInfo, ProcInfo
from .base import get_json

CACHE_TTL = 5.0


class OllamaResolver:
    name = "ollama"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._cache: list[dict] | None = None
        self._cache_ts = 0.0

    def running(self, force: bool = False) -> list[dict]:
        now = time.time()
        if not force and self._cache is not None and (now - self._cache_ts) < CACHE_TTL:
            return self._cache
        data = get_json(f"{self.cfg.ollama_url}/api/ps")
        models: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            models = [m for m in data["models"] if isinstance(m, dict)]
        self._cache = models
        self._cache_ts = now
        return models

    def matches(self, info: ProcInfo) -> bool:
        return info.name.lower().startswith("ollama")

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        if not self.matches(info):
            return None
        models = self.running()
        content = ContentInfo(
            kind="llm",
            backend="ollama",
            source=self.name,
            confidence=Confidence.CONFIRMED,
            endpoint=self.cfg.ollama_url,
        )
        if not models:
            content.kind = "app"
            content.label = "Ollama"
            content.detail = "nessun modello residente"
            return content

        primary = max(models, key=lambda m: int(m.get("size_vram") or 0))
        details = primary.get("details") or {}
        content.label = str(primary.get("name") or primary.get("model") or "modello Ollama")
        content.model_name = content.label
        content.quantization = str(details.get("quantization_level") or "") or None
        content.parameters = str(details.get("parameter_size") or "") or None
        content.file_size_bytes = int(primary.get("size") or 0) or None
        content.extra["size_vram"] = int(primary.get("size_vram") or 0)
        content.extra["family"] = details.get("family")
        content.extra["expires_at"] = primary.get("expires_at")
        content.extra["all_models"] = [
            {"name": m.get("name"), "size_vram": int(m.get("size_vram") or 0)}
            for m in models
        ]

        parts = [content.quantization, content.parameters]
        if len(models) > 1:
            parts.append(f"{len(models)} modelli residenti")
        ttl = _expires_in(primary.get("expires_at"))
        if ttl:
            parts.append(f"scade fra {ttl}")
        content.detail = " · ".join(p for p in parts if p)
        return content


def _expires_in(value: object) -> str | None:
    """Tempo rimanente prima dello scaricamento automatico, in forma leggibile."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    if delta <= 0:
        return None
    if delta < 90:
        return f"{int(delta)} s"
    if delta < 5400:
        return f"{int(delta / 60)} min"
    return f"{delta / 3600:.1f} h"
