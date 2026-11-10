"""Scaricare contenuto dalla VRAM.

Non esiste una API di sistema per liberare selettivamente l'allocazione di un
altro processo: e' il vincolo fondamentale del problema. Quello che si puo' fare
e' chiedere al processo di farlo, e la richiesta ha una scala di garbo:

  1. **API del backend** — l'unica via che libera davvero lasciando il processo
     vivo e riutilizzabile. LM Studio, Ollama, ComfyUI e A1111 espongono tutti un
     comando di scaricamento.
  2. **Agent** — per i processi PyTorch con l'agent attivo, `empty_cache()`
     restituisce al driver la memoria che l'allocatore di torch tiene in cache
     ma non usa. Non tocca i tensori vivi.
  3. **Trim / sospensione / terminazione** — in `proc_actions.py`, sempre su
     conferma esplicita.

Ogni azione riporta cosa e' stato fatto e quanta memoria e' stata liberata
davvero, misurata prima e dopo: dichiarare "fatto" senza verificare sarebbe
inutile per l'utente, che vuole sapere se ha risolto il problema.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.models import ContentInfo, ProcInfo
from ..core.units import human_bytes
from ..content.base import get_json, post_json

log = logging.getLogger("vramexp.unload")


@dataclass
class UnloadOption:
    """Un'azione di scaricamento proposta per un processo."""

    kind: str                 # backend_api | agent_empty_cache | trim | suspend | kill
    label: str                # testo del pulsante
    description: str          # cosa fa esattamente
    destructive: bool = False
    requires_confirm: bool = False
    available: bool = True
    unavailable_reason: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "label": self.label, "description": self.description,
            "destructive": self.destructive, "requires_confirm": self.requires_confirm,
            "available": self.available, "unavailable_reason": self.unavailable_reason,
            "params": self.params,
        }


@dataclass
class UnloadResult:
    ok: bool
    action: str
    message: str
    freed_bytes: int = 0
    before_bytes: int = 0
    after_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "action": self.action, "message": self.message,
            "freed_bytes": self.freed_bytes, "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "freed_human": human_bytes(self.freed_bytes),
        }


class Unloader:
    """Costruisce le opzioni disponibili per un processo ed esegue quella scelta."""

    def __init__(self, cfg: Config | None = None, agent_client=None,
                 free_bytes_fn=None) -> None:
        self.cfg = cfg or Config()
        self.agent_client = agent_client       # iniettato da F7
        # Funzione che misura la VRAM libera, per quantificare l'effetto reale.
        self._free_bytes = free_bytes_fn or _nvml_free_bytes

    # -- scoperta delle opzioni -------------------------------------------
    def options_for(self, info: ProcInfo) -> list[UnloadOption]:
        from .proc_actions import is_protected_process

        opts: list[UnloadOption] = []
        c = info.content

        backend_opt = self._backend_option(c)
        if backend_opt is not None:
            opts.append(backend_opt)

        if self.agent_client is not None and self.agent_client.has_agent(info.pid):
            opts.append(UnloadOption(
                kind="agent_empty_cache",
                label="Svuota la cache di PyTorch",
                description=("Restituisce al driver la memoria che l'allocatore di torch "
                             "tiene in cache senza usarla. I tensori vivi non vengono toccati "
                             "e il processo continua a funzionare."),
            ))

        protected, reason = is_protected_process(info)
        opts.append(UnloadOption(
            kind="trim",
            label="Riduci il working set",
            description=("Chiede a Windows di ridurre la memoria di lavoro del processo. "
                         "Effetto parziale e non garantito sulla VRAM."),
        ))
        opts.append(UnloadOption(
            kind="suspend",
            label="Sospendi il processo",
            description=("Congela il processo: l'OS puo' allora sfrattarne la memoria. "
                         "Reversibile con la ripresa, ma l'applicazione smette di rispondere."),
            destructive=True, requires_confirm=True,
            available=not protected, unavailable_reason=reason,
        ))
        opts.append(UnloadOption(
            kind="kill",
            label="Termina il processo",
            description=("Libera tutto immediatamente, ma il lavoro non salvato va perduto. "
                         "Ultima risorsa."),
            destructive=True, requires_confirm=True,
            available=not protected and self.cfg.allow_kill,
            unavailable_reason=reason if protected else (
                "" if self.cfg.allow_kill else "terminazione disabilitata nella configurazione"),
        ))
        return opts

    def _backend_option(self, c: ContentInfo) -> UnloadOption | None:
        backend = (c.backend or "").lower()

        if "lmstudio" in backend or (c.extra.get("lmstudio_id") and c.model_name):
            model_id = c.extra.get("lmstudio_id") or c.model_name
            return UnloadOption(
                kind="backend_api", label="Scarica il modello (LM Studio)",
                description=(f"Chiede a LM Studio di scaricare «{model_id}». "
                             "La VRAM viene liberata e il modello si puo' ricaricare quando serve."),
                params={"backend": "lmstudio", "model": model_id},
            )

        if backend == "ollama":
            model = c.model_name
            return UnloadOption(
                kind="backend_api", label="Scarica il modello (Ollama)",
                description=(f"Imposta keep_alive a 0 per «{model}», che lo rimuove "
                             "immediatamente dalla VRAM."),
                params={"backend": "ollama", "model": model},
            )

        if backend == "comfyui" and c.endpoint:
            return UnloadOption(
                kind="backend_api", label="Libera i modelli (ComfyUI)",
                description=("Chiede a ComfyUI di scaricare i modelli in cache e liberare "
                             "la memoria. Le generazioni in coda restano."),
                params={"backend": "comfyui", "endpoint": c.endpoint},
            )

        if backend == "a1111" and c.endpoint:
            return UnloadOption(
                kind="backend_api", label="Scarica il checkpoint (WebUI)",
                description="Rimuove il checkpoint dalla VRAM lasciando la WebUI attiva.",
                params={"backend": "a1111", "endpoint": c.endpoint},
            )

        if backend == "llama.cpp":
            # llama.cpp da solo non ha un endpoint di unload: il server tiene il
            # modello per tutta la sua vita. Si dichiara invece di fingere.
            return UnloadOption(
                kind="backend_api", label="Scarica il modello",
                description=("Il server llama.cpp avviato a mano non espone un comando di "
                             "scaricamento: tiene il modello finche' resta in esecuzione. "
                             "Per liberare la VRAM occorre terminare il processo."),
                available=False,
                unavailable_reason="llama.cpp non ha un endpoint di unload",
                params={"backend": "llamacpp"},
            )
        return None

    # -- esecuzione --------------------------------------------------------
    def execute(self, info: ProcInfo, kind: str, params: dict | None = None,
                confirmed: bool = False) -> UnloadResult:
        params = params or {}
        opt = next((o for o in self.options_for(info) if o.kind == kind), None)
        if opt is None:
            return UnloadResult(False, kind, f"azione «{kind}» non riconosciuta")
        if not opt.available:
            return UnloadResult(False, kind, opt.unavailable_reason or "azione non disponibile")
        if opt.requires_confirm and self.cfg.confirm_destructive and not confirmed:
            return UnloadResult(False, kind,
                                "azione distruttiva: richiede conferma esplicita")

        before = self._free_bytes()
        merged = {**opt.params, **params}

        if kind == "backend_api":
            ok, msg = self._call_backend(merged)
        elif kind == "agent_empty_cache":
            ok, msg = self._agent_empty_cache(info.pid)
        else:
            from .proc_actions import perform_process_action
            ok, msg = perform_process_action(info.pid, kind)

        # I backend rispondono prima di aver finito di liberare: si concede un
        # istante prima di misurare, altrimenti il risultato direbbe "0 byte".
        time.sleep(1.2)
        after = self._free_bytes()
        freed = max(0, after - before)
        return UnloadResult(ok=ok, action=kind, message=msg, freed_bytes=freed,
                            before_bytes=before, after_bytes=after)

    def _call_backend(self, p: dict) -> tuple[bool, str]:
        backend = p.get("backend")

        if backend == "lmstudio":
            model = p.get("model")
            if not model:
                return False, "modello non identificato"
            # L'API REST di LM Studio non espone lo scaricamento; la CLI `lms` si'.
            ok, msg = _run_lms_unload(str(model))
            return ok, msg

        if backend == "ollama":
            model = p.get("model")
            if not model:
                return False, "modello non identificato"
            ok, msg = post_json(f"{self.cfg.ollama_url}/api/generate",
                                {"model": model, "keep_alive": 0})
            return ok, ("scaricamento richiesto a Ollama" if ok else msg)

        if backend == "comfyui":
            endpoint = p.get("endpoint") or (self.cfg.comfyui_urls[0]
                                             if self.cfg.comfyui_urls else None)
            if not endpoint:
                return False, "endpoint ComfyUI sconosciuto"
            ok, msg = post_json(f"{endpoint}/free",
                                {"unload_models": True, "free_memory": True})
            return ok, ("modelli scaricati da ComfyUI" if ok else msg)

        if backend == "a1111":
            endpoint = p.get("endpoint")
            if not endpoint:
                return False, "endpoint WebUI sconosciuto"
            ok, msg = post_json(f"{endpoint}/sdapi/v1/unload-checkpoint", None)
            return ok, ("checkpoint scaricato" if ok else msg)

        return False, f"backend «{backend}» senza comando di scaricamento"

    def _agent_empty_cache(self, pid: int) -> tuple[bool, str]:
        if self.agent_client is None:
            return False, "agent non disponibile"
        try:
            resp = self.agent_client.call(pid, "empty_cache")
        except Exception as e:
            return False, f"chiamata all'agent fallita: {e}"
        if not resp or not resp.get("ok"):
            return False, str((resp or {}).get("error", "risposta non valida dall'agent"))
        freed = resp.get("freed_bytes")
        return True, (f"cache di torch svuotata ({human_bytes(freed)} restituiti al driver)"
                      if freed else "cache di torch svuotata")


def _run_lms_unload(model: str) -> tuple[bool, str]:
    """Scarica un modello tramite la CLI `lms`.

    LM Studio non espone lo scaricamento via REST; la CLI si', ed e' installata
    insieme all'applicazione.
    """
    last = "nessun tentativo eseguito"
    # Si prova prima con l'identificatore preciso; `--all` e' il ripiego per
    # quando l'id noto a noi non coincide con quello atteso dalla CLI.
    for argv in (["lms", "unload", model], ["lms", "unload", "--all"]):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=30, creationflags=_no_window())
        except FileNotFoundError:
            return False, ("CLI `lms` non trovata: installarla da LM Studio "
                           "(Settings → Developer → Install CLI)")
        except subprocess.TimeoutExpired:
            return False, "la CLI `lms` non ha risposto entro 30 secondi"
        if proc.returncode == 0:
            scope = model if argv[-1] == model else "tutti i modelli"
            return True, f"scaricato {scope} tramite lms"
        last = (proc.stderr or proc.stdout or "").strip()[:300]
    return False, f"lms unload fallito: {last}"


def _no_window() -> int:
    """Evita che le CLI facciano lampeggiare una finestra di console."""
    import sys
    return 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW


def _nvml_free_bytes() -> int:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(h).free)
    except Exception:
        return 0
