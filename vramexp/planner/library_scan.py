"""Scansione delle librerie di modelli su disco.

Serve al planner: per rispondere a "ci sta un 70B?" bisogna prima sapere quali
modelli esistono e quanto pesano. Copre i formati che occupano VRAM in modo
prevedibile: GGUF (con metadati completi) e safetensors/checkpoint (dove si puo'
dedurre solo la dimensione, ma basta per il calcolo dei pesi).

La scansione e' incrementale e mantiene una cache su disco indicizzata per
(percorso, dimensione, data di modifica): rileggere l'header di duecento GGUF a
ogni avvio costerebbe secondi e non cambierebbe nulla nella quasi totalita' dei
casi.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.config import app_dir
from ..content.gguf_header import GgufInfo, read_gguf

log = logging.getLogger("vramexp.library")

CACHE_PATH = app_dir() / "model_library.json"
MODEL_SUFFIXES = {".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
MIN_SIZE = 32 * 1024 * 1024      # sotto questa soglia non sono modelli interessanti


@dataclass
class ModelEntry:
    path: str
    name: str
    size_bytes: int
    mtime: float
    fmt: str                       # gguf | safetensors | checkpoint
    architecture: str | None = None
    quantization: str | None = None
    parameters: str | None = None
    n_layers: int | None = None
    n_kv_heads: int | None = None
    n_heads: int | None = None
    head_dim: int | None = None
    context_length: int | None = None
    error: str | None = None

    @property
    def key(self) -> str:
        """Identificativo stabile del modello, per correlare le osservazioni
        storiche anche se il file viene spostato."""
        return f"{self.name}|{self.size_bytes}"

    def to_gguf_info(self) -> GgufInfo:
        """Adatta la voce al formato atteso dallo stimatore."""
        g = GgufInfo(path=self.path, file_size=self.size_bytes)
        g.architecture = self.architecture
        g.quantization = self.quantization
        g.parameters = self.parameters
        g.n_layers = self.n_layers
        g.n_heads = self.n_heads
        g.n_kv_heads = self.n_kv_heads
        g.key_length = self.head_dim
        g.context_length = self.context_length
        return g

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d


@dataclass
class LibraryIndex:
    models: list[ModelEntry] = field(default_factory=list)
    scanned_at: float = 0.0
    dirs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def total_bytes(self) -> int:
        return sum(m.size_bytes for m in self.models)

    def find(self, needle: str) -> list[ModelEntry]:
        key = needle.lower().replace("-", "").replace("_", "").replace(" ", "")
        out = []
        for m in self.models:
            name = m.name.lower().replace("-", "").replace("_", "").replace(" ", "")
            if key in name:
                out.append(m)
        return sorted(out, key=lambda m: -m.size_bytes)

    def to_dict(self) -> dict:
        return {
            "models": [m.to_dict() for m in self.models],
            "scanned_at": self.scanned_at,
            "dirs": self.dirs,
            "errors": self.errors,
            "count": len(self.models),
            "total_bytes": self.total_bytes(),
        }


class LibraryScanner:
    def __init__(self, dirs: list[str]) -> None:
        self.dirs = [d for d in dirs if d]
        self._cache: dict[str, dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not CACHE_PATH.exists():
            return
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            self._cache = {m["path"]: m for m in data.get("models", [])}
        except (OSError, json.JSONDecodeError, KeyError):
            self._cache = {}

    def _save_cache(self, index: LibraryIndex) -> None:
        try:
            CACHE_PATH.write_text(
                json.dumps(index.to_dict(), ensure_ascii=False, indent=1),
                encoding="utf-8")
        except OSError as e:
            log.debug("cache libreria non salvata: %s", e)

    def scan(self, deep: bool = True, max_files: int = 5000) -> LibraryIndex:
        """Percorre le directory configurate e indicizza i modelli trovati."""
        index = LibraryIndex(dirs=list(self.dirs), scanned_at=time.time())
        seen: set[str] = set()

        for root_dir in self.dirs:
            root = Path(root_dir)
            if not root.exists():
                index.errors.append(f"percorso inesistente: {root_dir}")
                continue
            try:
                for path in root.rglob("*"):
                    if len(seen) >= max_files:
                        index.errors.append(
                            f"raggiunto il limite di {max_files} file: scansione parziale")
                        break
                    if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    if stat.st_size < MIN_SIZE:
                        continue
                    key = str(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    index.models.append(self._entry(path, stat, deep))
            except (OSError, PermissionError) as e:
                index.errors.append(f"{root_dir}: {e}")

        index.models.sort(key=lambda m: -m.size_bytes)
        self._save_cache(index)
        return index

    def _entry(self, path: Path, stat: os.stat_result, deep: bool) -> ModelEntry:
        cached = self._cache.get(str(path))
        # La cache vale finche' dimensione e data di modifica coincidono: e' un
        # controllo sufficiente per file di modello, che non vengono modificati
        # in place.
        if (cached and cached.get("size_bytes") == stat.st_size
                and abs(float(cached.get("mtime", 0)) - stat.st_mtime) < 1.0):
            cached = {k: v for k, v in cached.items() if k != "key"}
            try:
                return ModelEntry(**cached)
            except TypeError:
                pass   # formato della cache cambiato: si rilegge

        suffix = path.suffix.lower()
        fmt = "gguf" if suffix == ".gguf" else (
            "safetensors" if suffix == ".safetensors" else "checkpoint")
        entry = ModelEntry(path=str(path), name=path.stem, size_bytes=stat.st_size,
                           mtime=stat.st_mtime, fmt=fmt)

        if fmt == "gguf" and deep:
            g = read_gguf(path)
            if g is not None:
                entry.architecture = g.architecture
                entry.quantization = g.quantization
                entry.parameters = g.parameters
                entry.n_layers = g.n_layers
                entry.n_heads = g.n_heads
                entry.n_kv_heads = g.n_kv_heads
                entry.head_dim = g.head_dim
                entry.context_length = g.context_length
            else:
                entry.error = "header GGUF illeggibile"
        return entry
