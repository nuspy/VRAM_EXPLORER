"""LM Studio: modelli caricati via API REST locale.

LM Studio non tiene i modelli nel proprio processo: avvia un `llama-server`
figlio per ognuno. Questo resolver serve quindi a due scopi diversi:

  * etichettare il processo `LM Studio.exe` (che occupa VRAM solo per la propria
    interfaccia Electron);
  * fornire al resolver di llama.cpp la conferma autorevole di *quale* modello
    e' caricato, con tipo e quantizzazione dichiarati dal backend invece che
    dedotti dal nome del file.

L'endpoint `/api/v0/models` e' specifico di LM Studio ed e' piu' ricco di
`/v1/models`: riporta `state` (loaded/not-loaded), `max_context_length`,
`quantization` e la distinzione llm / vlm / embeddings.
"""

from __future__ import annotations

import time

from ..core.config import Config
from ..core.models import Confidence, ContentInfo, ProcInfo
from .base import get_json

CACHE_TTL = 10.0


class LmStudioResolver:
    name = "lmstudio"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._cache: list[dict] | None = None
        self._cache_ts = 0.0

    # -- API ---------------------------------------------------------------
    def models(self, force: bool = False) -> list[dict]:
        """Elenco dei modelli noti a LM Studio, con cache breve.

        La cache evita di martellare l'API una volta per ogni processo figlio
        durante lo stesso ciclo di campionamento.
        """
        now = time.time()
        if not force and self._cache is not None and (now - self._cache_ts) < CACHE_TTL:
            return self._cache
        data = get_json(f"{self.cfg.lmstudio_url}/api/v0/models")
        models: list[dict] = []
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            models = [m for m in data["data"] if isinstance(m, dict)]
        self._cache = models
        self._cache_ts = now
        return models

    def loaded_models(self) -> list[dict]:
        return [m for m in self.models() if m.get("state") == "loaded"]

    def find_model(self, needle: str) -> dict | None:
        """Cerca un modello per id o per frammento di percorso.

        Il confronto e' normalizzato perche' l'id LM Studio
        ("qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive") e il nome del file
        GGUF differiscono per separatori e maiuscole.
        """
        if not needle:
            return None
        key = _norm(needle)
        best: dict | None = None
        for m in self.models():
            mid = _norm(str(m.get("id", "")))
            if not mid:
                continue
            if mid == key:
                return m
            if (mid in key or key in mid) and (best is None or len(mid) > len(_norm(str(best.get("id", ""))))):
                best = m
        return best

    # -- resolver ----------------------------------------------------------
    def matches(self, info: ProcInfo) -> bool:
        # Match sul solo nome dell'eseguibile, mai sul percorso: i backend
        # llama-server risiedono sotto `.lmstudio\extensions\` e verrebbero
        # scambiati per l'interfaccia, perdendo l'identificazione del modello.
        low = info.name.lower()
        return low in ("lm studio.exe", "lms.exe", "lm studio")

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        if not self.matches(info):
            return None
        loaded = self.loaded_models()
        content = ContentInfo(
            kind="app",
            backend="lmstudio",
            source=self.name,
            confidence=Confidence.CONFIRMED,
            endpoint=self.cfg.lmstudio_url,
            label="LM Studio",
        )
        if loaded:
            names = ", ".join(str(m.get("id", "?")) for m in loaded[:3])
            content.detail = f"interfaccia · {len(loaded)} modello/i caricato/i: {names}"
            content.extra["loaded_models"] = [m.get("id") for m in loaded]
        else:
            content.detail = "interfaccia · nessun modello caricato"
        content.extra["note"] = (
            "i modelli girano in processi llama-server separati, non qui"
        )
        return content

    def enrich(self, content: ContentInfo) -> None:
        """Arricchisce un contenuto llama.cpp con i dati dichiarati da LM Studio.

        Serve a promuovere la confidenza da PARSED a CONFIRMED e ad aggiungere il
        tipo (vlm vs llm), che dal nome del file non e' deducibile.
        """
        needle = content.extra.get("served_id") or content.model_name or ""
        m = self.find_model(str(needle))
        if not m:
            return
        content.extra["lmstudio_id"] = m.get("id")
        kind = str(m.get("type") or "")
        if kind in ("llm", "vlm", "embeddings"):
            content.kind = kind
        if m.get("quantization") and not content.quantization:
            content.quantization = str(m["quantization"])
        if m.get("max_context_length") and not content.context_length:
            content.context_length = int(m["max_context_length"])
        if m.get("state") == "loaded":
            content.confidence = Confidence.CONFIRMED
        content.backend = "lmstudio/llama.cpp"


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())
