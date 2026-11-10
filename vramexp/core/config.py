"""Configurazione persistente, in %LOCALAPPDATA%\\VramExplorer\\config.json.

Le impostazioni con implicazioni di sicurezza (attach a caldo, azioni distruttive)
sono disattivate per default e devono essere abilitate esplicitamente: il
servizio e' stato progettato in sola osservazione, e la configurazione deve
riflettere quella scelta anche se qualcuno modifica il file a mano.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    d = Path(base) / "VramExplorer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    d = app_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


CONFIG_PATH = app_dir() / "config.json"
DB_PATH = app_dir() / "history.db"
TOKEN_PATH = app_dir() / "api_token.txt"
PROFILES_DIR = app_dir() / "profiles"


@dataclass
class Config:
    # --- campionamento ---
    sample_interval_s: float = 1.0
    identity_refresh_s: float = 5.0      # ogni quanto rivalutare cmdline/contenuto
    content_refresh_s: float = 15.0      # ogni quanto ri-interrogare le API dei backend

    # --- API ---
    api_host: str = "127.0.0.1"          # loopback only: non esporre mai in rete
    api_port: int = 8765
    api_require_token: bool = True

    # --- storico ---
    retain_full_hours: float = 1.0       # 1 s di risoluzione
    retain_medium_hours: float = 24.0    # 10 s
    retain_coarse_days: float = 30.0     # 60 s
    vacuum_interval_h: float = 24.0

    # --- rilevamento (osservazione: notifica, non agisce) ---
    warn_used_pct: float = 90.0
    critical_used_pct: float = 97.0
    spillover_warn_bytes: int = 64 * 1024 * 1024
    oom_suspect_used_pct: float = 85.0   # sotto questa soglia una morte non e' sospetta
    min_track_bytes: int = 16 * 1024 * 1024  # ignora processi sotto questa soglia

    # --- azioni ---
    confirm_destructive: bool = True     # kill/suspend richiedono conferma esplicita
    allow_kill: bool = True
    enable_hot_attach: bool = False      # injection a caldo: puo' far crashare il target
    balloon_chunk_bytes: int = 256 * 1024 * 1024

    # --- endpoint dei backend da interrogare ---
    lmstudio_url: str = "http://127.0.0.1:1234"
    ollama_url: str = "http://127.0.0.1:11434"
    comfyui_urls: list[str] = field(default_factory=lambda: [
        "http://127.0.0.1:8188", "http://127.0.0.1:8189",
    ])
    a1111_urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:7860"])
    extra_openai_urls: list[str] = field(default_factory=list)

    # --- librerie di modelli per il planner ---
    model_dirs: list[str] = field(default_factory=lambda: [
        r"E:\AI\Models",
        str(Path.home() / ".lmstudio" / "models"),
        str(Path.home() / ".ollama" / "models"),
    ])

    # --- GUI ---
    theme: str = "dark"
    start_minimized: bool = False
    show_system_processes: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        p = path or CONFIG_PATH
        cfg = cls()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                # Config corrotta: proseguire con i default e dirlo, invece di
                # impedire l'avvio del servizio per un file malformato.
                print(f"[config] {p} illeggibile ({e}); uso i valori predefiniti")
                return cfg
            known = {f.name for f in fields(cls)}
            for k, v in raw.items():
                if k in known:
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: Path | None = None) -> None:
        p = path or CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(p)   # scrittura atomica: mai una config mezza scritta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_or_create_token() -> str:
    """Token dell'API locale. Generato al primo avvio e riusato in seguito."""
    if TOKEN_PATH.exists():
        tok = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    TOKEN_PATH.write_text(tok, encoding="utf-8")
    try:
        os.chmod(TOKEN_PATH, 0o600)
    except OSError:
        pass  # su Windows le ACL non seguono il modello POSIX; il file resta nel profilo utente
    return tok
