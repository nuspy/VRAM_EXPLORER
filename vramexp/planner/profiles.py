"""Profili operativi: riconfigurare la macchina con un'azione sola.

Un profilo e' una sequenza dichiarativa di passi — scaricare modelli, impostare
la riserva, cambiare la policy di fallback. Serve a rendere ripetibile un
passaggio di contesto che altrimenti richiede otto click sparsi: "smetto di
lavorare sugli LLM e mi metto a giocare" oppure "libero tutto per un training".

Regola di progetto, coerente con la scelta di un servizio in sola osservazione:
**nessun profilo si applica senza anteprima**. `preview()` produce l'elenco
esatto delle azioni con la memoria che ognuna dovrebbe liberare, e solo dopo
`apply()` puo' essere invocato. Un pulsante che riconfigura la macchina senza
dire cosa sta per fare sarebbe pericoloso in proporzione alla sua comodita'.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import PROFILES_DIR
from ..core.units import human_bytes, parse_size

log = logging.getLogger("vramexp.profiles")

STEP_KINDS = {
    "unload_backend": "Scarica i modelli di un backend",
    "unload_process": "Scarica il contenuto di un processo",
    "kill_process": "Termina un processo",
    "set_reserve": "Imposta la riserva di VRAM",
    "set_fallback_policy": "Imposta la policy di fallback",
    "empty_cache": "Svuota la cache di PyTorch dei processi con agent",
}


@dataclass
class ProfileStep:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Profile:
    name: str
    title: str
    description: str = ""
    steps: list[ProfileStep] = field(default_factory=list)
    builtin: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "title": self.title,
                "description": self.description, "builtin": self.builtin,
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            name=str(d.get("name", "")),
            title=str(d.get("title", d.get("name", ""))),
            description=str(d.get("description", "")),
            builtin=bool(d.get("builtin", False)),
            steps=[ProfileStep(kind=str(s.get("kind", "")),
                               params=dict(s.get("params") or {}),
                               description=str(s.get("description", "")))
                   for s in (d.get("steps") or [])],
        )


BUILTIN_PROFILES = [
    Profile(
        name="gaming", title="Gioco", builtin=True,
        description=("Libera tutta la VRAM occupata dai carichi di lavoro AI e "
                     "ripristina il comportamento permissivo del driver, che per i "
                     "giochi è preferibile a un crash."),
        steps=[
            ProfileStep("unload_backend", {"backend": "lmstudio"},
                        "Scarica i modelli caricati in LM Studio"),
            ProfileStep("unload_backend", {"backend": "ollama"},
                        "Scarica i modelli residenti in Ollama"),
            ProfileStep("unload_backend", {"backend": "comfyui"},
                        "Libera i modelli di ComfyUI"),
            ProfileStep("empty_cache", {},
                        "Svuota la cache torch nei processi con agent attivo"),
            ProfileStep("set_reserve", {"bytes": 0},
                        "Azzera la riserva di VRAM"),
            ProfileStep("set_fallback_policy", {"value": 0},
                        "Consenti l'uso della RAM: per un gioco il calo di prestazioni "
                        "è preferibile a un blocco"),
        ],
    ),
    Profile(
        name="llm", title="Inferenza LLM", builtin=True,
        description=("Prepara la macchina a ospitare un modello linguistico grande: "
                     "libera i carichi grafici e impone il fail-fast a Python, così un "
                     "contesto sovradimensionato fallisce subito invece di degradare."),
        steps=[
            ProfileStep("unload_backend", {"backend": "comfyui"},
                        "Libera i modelli di diffusione"),
            ProfileStep("unload_backend", {"backend": "a1111"},
                        "Scarica il checkpoint della WebUI"),
            ProfileStep("set_reserve", {"bytes": 0},
                        "Rilascia la riserva per lasciare spazio al modello"),
            ProfileStep("set_fallback_policy", {"value": 1, "exe": "python.exe"},
                        "Nessun fallback su RAM per python.exe"),
        ],
    ),
    Profile(
        name="training", title="Addestramento", builtin=True,
        description=("Massimo spazio contiguo e fallimento immediato in caso di "
                     "memoria insufficiente: su un training lungo un errore subito "
                     "vale molto più di un rallentamento scoperto ore dopo."),
        steps=[
            ProfileStep("unload_backend", {"backend": "lmstudio"},
                        "Scarica i modelli di LM Studio"),
            ProfileStep("unload_backend", {"backend": "comfyui"},
                        "Libera i modelli di ComfyUI"),
            ProfileStep("empty_cache", {}, "Svuota le cache torch"),
            ProfileStep("set_fallback_policy", {"value": 1, "exe": "python.exe"},
                        "Nessun fallback su RAM per python.exe"),
        ],
    ),
    Profile(
        name="reserve_workspace", title="Riserva spazio di lavoro", builtin=True,
        description=("Trattiene 8 GB perché restino disponibili quando servono, "
                     "impedendo ad altri processi CUDA di occuparli mentre non guardi."),
        steps=[
            ProfileStep("set_reserve", {"bytes": 8 * 1024 ** 3},
                        "Riserva 8 GB di VRAM"),
        ],
    ),
]


@dataclass
class PreviewItem:
    step: ProfileStep
    title: str
    detail: str
    applicable: bool
    estimated_free_bytes: int = 0
    warning: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.step.kind, "title": self.title, "detail": self.detail,
                "applicable": self.applicable,
                "estimated_free_bytes": self.estimated_free_bytes,
                "estimated_free_human": (human_bytes(self.estimated_free_bytes)
                                         if self.estimated_free_bytes else ""),
                "warning": self.warning, "params": self.step.params}


class ProfileManager:
    """Carica, salva, mostra in anteprima e applica i profili."""

    def __init__(self, unloader=None, balloon_ctl=None, policy=None,
                 agent_client=None) -> None:
        self.unloader = unloader
        self.balloon_ctl = balloon_ctl
        self.policy = policy
        self.agent_client = agent_client
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # -- persistenza -------------------------------------------------------
    def list(self) -> list[Profile]:
        profiles = list(BUILTIN_PROFILES)
        names = {p.name for p in profiles}
        for path in sorted(PROFILES_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                p = Profile.from_dict(data)
                if not p.name:
                    continue
                # Un profilo utente con lo stesso nome sostituisce il predefinito:
                # cosi' si puo' adattare "gaming" senza perdere il resto.
                profiles = [x for x in profiles if x.name != p.name]
                profiles.append(p)
                names.add(p.name)
            except (OSError, json.JSONDecodeError) as e:
                log.warning("profilo %s illeggibile: %s", path.name, e)
        return profiles

    def get(self, name: str) -> Profile | None:
        return next((p for p in self.list() if p.name == name), None)

    def save(self, profile: Profile) -> Path:
        profile.builtin = False
        path = PROFILES_DIR / f"{_safe_name(profile.name)}.json"
        path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def delete(self, name: str) -> bool:
        path = PROFILES_DIR / f"{_safe_name(name)}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    # -- anteprima ---------------------------------------------------------
    def preview(self, name: str, snapshot) -> dict:
        """Cosa farebbe il profilo, senza fare nulla."""
        profile = self.get(name)
        if profile is None:
            return {"ok": False, "error": f"profilo «{name}» inesistente"}

        items: list[PreviewItem] = []
        total_free = 0
        for step in profile.steps:
            item = self._preview_step(step, snapshot)
            items.append(item)
            if item.applicable:
                total_free += item.estimated_free_bytes

        return {
            "ok": True,
            "profile": profile.to_dict(),
            "items": [i.to_dict() for i in items],
            "estimated_total_free_bytes": total_free,
            "estimated_total_free_human": human_bytes(total_free),
            "applicable_steps": sum(1 for i in items if i.applicable),
            "total_steps": len(items),
        }

    def _preview_step(self, step: ProfileStep, snapshot) -> PreviewItem:
        title = STEP_KINDS.get(step.kind, step.kind)
        detail = step.description

        if step.kind in ("unload_backend", "unload_process", "kill_process",
                         "empty_cache"):
            targets = self._targets_for(step, snapshot)
            if not targets:
                return PreviewItem(step, title, detail, applicable=False,
                                   warning="nessun processo corrispondente in esecuzione")
            freed = sum(t.sample.dedicated_bytes for t in targets)
            names = ", ".join(f"{t.info.content.label or t.info.name} (pid {t.info.pid})"
                              for t in targets[:3])
            warn = ""
            if step.kind == "kill_process":
                warn = "termina il processo: il lavoro non salvato va perduto"
            return PreviewItem(step, title, f"{detail} → {names}", True, freed, warn)

        if step.kind == "set_reserve":
            target = _as_bytes(step.params.get("bytes", 0))
            current = getattr(snapshot, "balloon_bytes", 0)
            return PreviewItem(
                step, title,
                f"riserva da {human_bytes(current)} a {human_bytes(target)}",
                applicable=(target != current),
                estimated_free_bytes=max(0, current - target),
                warning=("la riserva è best-effort: non impedisce al compositor "
                         "di ottenere memoria" if target else ""))

        if step.kind == "set_fallback_policy":
            value = int(step.params.get("value", 0))
            exe = step.params.get("exe")
            scope = f"per {exe}" if exe else "globalmente"
            label = ("nessun fallback su RAM" if value else
                     "consenti l'uso della RAM")
            return PreviewItem(
                step, title, f"{label} {scope}", applicable=True,
                warning="richiede l'elevazione: comparirà un prompt UAC")

        return PreviewItem(step, title, detail, applicable=False,
                           warning=f"tipo di passo sconosciuto: {step.kind}")

    def _targets_for(self, step: ProfileStep, snapshot) -> list:
        backend = str(step.params.get("backend", "")).lower()
        pid = step.params.get("pid")
        out = []
        for entry in snapshot.processes:
            if pid is not None:
                if entry.info.pid == int(pid):
                    out.append(entry)
                continue
            if step.kind == "empty_cache":
                if self.agent_client is not None and self.agent_client.has_agent(entry.info.pid):
                    out.append(entry)
                continue
            if backend and backend in (entry.info.content.backend or "").lower():
                out.append(entry)
        return out

    # -- applicazione ------------------------------------------------------
    def apply(self, name: str, snapshot, confirmed: bool = False) -> dict:
        """Esegue il profilo. Richiede la conferma successiva all'anteprima."""
        if not confirmed:
            return {"ok": False,
                    "error": ("applicare un profilo richiede conferma esplicita: "
                              "usare prima l'anteprima")}
        profile = self.get(name)
        if profile is None:
            return {"ok": False, "error": f"profilo «{name}» inesistente"}

        results: list[dict] = []
        for step in profile.steps:
            results.append(self._apply_step(step, snapshot))

        succeeded = sum(1 for r in results if r.get("ok"))
        return {
            "ok": True, "profile": profile.name,
            "steps": results,
            "succeeded": succeeded, "total": len(results),
            "summary": f"{succeeded} passi su {len(results)} completati",
        }

    def _apply_step(self, step: ProfileStep, snapshot) -> dict:
        kind = step.kind
        try:
            if kind in ("unload_backend", "unload_process"):
                targets = self._targets_for(step, snapshot)
                if not targets:
                    return {"kind": kind, "ok": True, "skipped": True,
                            "message": "nessun processo corrispondente"}
                messages = []
                for t in targets:
                    res = self.unloader.execute(t.info, "backend_api", confirmed=True)
                    messages.append(f"pid {t.info.pid}: {res.message}")
                return {"kind": kind, "ok": True, "message": "; ".join(messages)}

            if kind == "empty_cache":
                targets = self._targets_for(step, snapshot)
                messages = []
                for t in targets:
                    res = self.unloader.execute(t.info, "agent_empty_cache", confirmed=True)
                    messages.append(f"pid {t.info.pid}: {res.message}")
                return {"kind": kind, "ok": True,
                        "message": "; ".join(messages) or "nessun processo con agent"}

            if kind == "kill_process":
                targets = self._targets_for(step, snapshot)
                messages = []
                for t in targets:
                    res = self.unloader.execute(t.info, "kill", confirmed=True)
                    messages.append(f"pid {t.info.pid}: {res.message}")
                return {"kind": kind, "ok": True, "message": "; ".join(messages)}

            if kind == "set_reserve":
                target = _as_bytes(step.params.get("bytes", 0))
                state = self.balloon_ctl.set_reserve(target)
                return {"kind": kind, "ok": True,
                        "message": f"riserva impostata a {human_bytes(state.get('allocated_bytes', 0))}"}

            if kind == "set_fallback_policy":
                ok, msg = self.policy.set(int(step.params.get("value", 0)),
                                          step.params.get("exe"))
                return {"kind": kind, "ok": ok, "message": msg}

            return {"kind": kind, "ok": False, "message": f"tipo sconosciuto: {kind}"}
        except Exception as e:
            return {"kind": kind, "ok": False, "message": f"{type(e).__name__}: {e}"}


def _as_bytes(value: Any) -> int:
    if isinstance(value, str):
        return parse_size(value)
    return int(value or 0)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:64]
