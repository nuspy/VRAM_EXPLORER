"""Pianificatore: "ci sta questo modello, e se no cosa devo liberare?"

Mette insieme lo stato attuale della VRAM, la stima del fabbisogno e le opzioni
di scaricamento disponibili, per rispondere con un si'/no motivato **e un piano
d'azione**. Una risposta "no" senza indicazioni su cosa fare sarebbe la meta' del
lavoro.

Due raffinamenti rispetto al puro calcolo:

* se lo storico contiene una misura reale dello stesso modello, viene preferita
  alla stima teorica — dopo qualche caricamento lo strumento smette di stimare e
  comincia a sapere;
* i candidati da liberare sono ordinati per rapporto fra memoria recuperabile e
  costo dell'operazione: scaricare un modello inattivo viene prima che terminare
  un processo che sta lavorando.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.models import Snapshot
from ..core.units import human_bytes
from .kv_cache import VramEstimate, estimate_gguf, max_context_for_budget
from .library_scan import ModelEntry


@dataclass
class FreeCandidate:
    pid: int
    label: str
    process_name: str
    recoverable_bytes: int
    action_kind: str
    action_label: str
    cost: str            # "basso" | "medio" | "alto"
    idle: bool = False

    def to_dict(self) -> dict:
        return {
            "pid": self.pid, "label": self.label, "process_name": self.process_name,
            "recoverable_bytes": self.recoverable_bytes,
            "recoverable_human": human_bytes(self.recoverable_bytes),
            "action_kind": self.action_kind, "action_label": self.action_label,
            "cost": self.cost, "idle": self.idle,
        }


@dataclass
class PlanResult:
    fits: bool
    verdict: str
    required_bytes: int
    free_bytes: int
    total_bytes: int
    shortfall_bytes: int = 0
    estimate: VramEstimate | None = None
    candidates: list[FreeCandidate] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    max_context_now: int = 0
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fits": self.fits, "verdict": self.verdict,
            "required_bytes": self.required_bytes, "required_human": human_bytes(self.required_bytes),
            "free_bytes": self.free_bytes, "free_human": human_bytes(self.free_bytes),
            "total_bytes": self.total_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "shortfall_human": human_bytes(self.shortfall_bytes) if self.shortfall_bytes else "",
            "estimate": self.estimate.to_dict() if self.estimate else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "plan_steps": self.plan_steps,
            "max_context_now": self.max_context_now,
            "alternatives": self.alternatives,
        }


class Planner:
    def __init__(self, store=None, unloader=None) -> None:
        self.store = store        # per le osservazioni storiche
        self.unloader = unloader  # per sapere quali azioni sono davvero possibili

    def plan(self, model: ModelEntry, snapshot: Snapshot, context: int | None = None,
             gpu_index: int = 0, gpu_layers: int | None = None,
             kv_type: str = "f16") -> PlanResult:
        gpu = next((g for g in snapshot.gpus if g.gpu_index == gpu_index), None)
        if gpu is None:
            return PlanResult(False, "GPU non trovata", 0, 0, 0)

        estimate = estimate_gguf(model.to_gguf_info(), context=context,
                                 gpu_layers=gpu_layers, kv_type=kv_type)
        required = estimate.total_bytes

        observed = self._observed_bytes(model, context)
        if observed:
            estimate.is_estimate = False
            estimate.confidence_note = (
                f"misura reale di un caricamento precedente ({human_bytes(observed)}), "
                "usata al posto della stima")
            required = observed

        free = gpu.free_bytes
        fits = required <= free
        result = PlanResult(
            fits=fits,
            verdict="",
            required_bytes=required,
            free_bytes=free,
            total_bytes=gpu.total_bytes,
            estimate=estimate,
            max_context_now=max_context_for_budget(model.to_gguf_info(), free, kv_type),
        )

        if fits:
            margin = free - required
            result.verdict = (
                f"Sì: servono {human_bytes(required)} e ce ne sono liberi "
                f"{human_bytes(free)} (margine {human_bytes(margin)}).")
            if margin < 1024 ** 3:
                result.verdict += (" Il margine è stretto: il caricamento potrebbe "
                                   "riuscire e poi degradare quando la cache KV si riempie.")
            return result

        result.shortfall_bytes = required - free
        result.verdict = (
            f"No: servono {human_bytes(required)} ma ce ne sono liberi solo "
            f"{human_bytes(free)}. Mancano {human_bytes(result.shortfall_bytes)}.")

        result.candidates = self._candidates(snapshot, gpu_index)
        result.plan_steps = self._steps(result.candidates, result.shortfall_bytes)
        result.alternatives = self._alternatives(model, free, estimate, context, kv_type)
        return result

    # -- candidati da liberare --------------------------------------------
    def _candidates(self, snapshot: Snapshot, gpu_index: int) -> list[FreeCandidate]:
        out: list[FreeCandidate] = []
        for entry in snapshot.processes:
            if entry.sample.gpu_index != gpu_index:
                continue
            # Sotto i 128 MB il recupero non sposta l'esito e affollerebbe la lista.
            if entry.sample.dedicated_bytes < 128 * 1024 * 1024:
                continue

            info = entry.info
            content = info.content
            options = self.unloader.options_for(info) if self.unloader else []
            backend = next((o for o in options
                            if o.kind == "backend_api" and o.available), None)
            idle = self._is_idle(entry)

            if backend is not None:
                out.append(FreeCandidate(
                    pid=info.pid, label=content.label or info.name,
                    process_name=info.name,
                    recoverable_bytes=entry.sample.dedicated_bytes,
                    action_kind="backend_api", action_label=backend.label,
                    cost="basso", idle=idle))
                continue

            kill = next((o for o in options if o.kind == "kill" and o.available), None)
            if kill is not None:
                out.append(FreeCandidate(
                    pid=info.pid, label=content.label or info.name,
                    process_name=info.name,
                    recoverable_bytes=entry.sample.dedicated_bytes,
                    action_kind="kill", action_label=kill.label,
                    cost="alto", idle=idle))

        # Prima il recupero facile, poi quello grande: scaricare un modello
        # inattivo e' sempre preferibile a terminare un processo che sta lavorando.
        cost_rank = {"basso": 0, "medio": 1, "alto": 2}
        out.sort(key=lambda c: (cost_rank.get(c.cost, 3), not c.idle,
                                -c.recoverable_bytes))
        return out

    def _is_idle(self, entry) -> bool:
        engines = entry.sample.engine_pct or {}
        return all(v < 2.0 for v in engines.values()) if engines else True

    def _steps(self, candidates: list[FreeCandidate], needed: int) -> list[str]:
        """Sequenza minima di azioni che libera abbastanza memoria."""
        steps: list[str] = []
        recovered = 0
        for c in candidates:
            if recovered >= needed:
                break
            steps.append(
                f"{c.action_label} — «{c.label}» (pid {c.pid}): "
                f"libera circa {human_bytes(c.recoverable_bytes)}"
                + (" · inattivo" if c.idle else " · in uso, attenzione"))
            recovered += c.recoverable_bytes
        if recovered < needed:
            steps.append(
                f"Anche liberando tutto il liberabile mancherebbero "
                f"{human_bytes(needed - recovered)}: valutare una quantizzazione "
                "più aggressiva o un contesto più corto.")
        return steps

    def _alternatives(self, model: ModelEntry, free: int, estimate: VramEstimate,
                      context: int | None, kv_type: str) -> list[str]:
        """Modi di far entrare il modello senza liberare nulla."""
        alts: list[str] = []
        gguf = model.to_gguf_info()

        max_ctx = max_context_for_budget(gguf, free, kv_type)
        if max_ctx > 0 and context and max_ctx < context:
            alts.append(
                f"Con il contesto ridotto a {max_ctx:,} token entrerebbe "
                "senza liberare nulla.".replace(",", " "))

        if estimate.kv_cache_bytes > 512 * 1024 * 1024 and kv_type == "f16":
            saving = estimate.kv_cache_bytes - int(estimate.kv_cache_bytes * 0.53)
            alts.append(
                f"Quantizzando la cache KV a q8_0 (-ctk q8_0 -ctv q8_0) si "
                f"risparmiano circa {human_bytes(saving)}, con perdita di qualità minima.")

        if gguf.n_layers:
            per_layer = model.size_bytes / gguf.n_layers
            if per_layer > 0:
                fit_layers = int(max(0, free - estimate.overhead_bytes
                                     - estimate.kv_cache_bytes) / per_layer)
                if 0 < fit_layers < gguf.n_layers:
                    alts.append(
                        f"Con -ngl {fit_layers} entrerebbero {fit_layers} layer su "
                        f"{gguf.n_layers}; i restanti girerebbero sulla CPU, "
                        "riducendo molto la velocità.")
        return alts

    def _observed_bytes(self, model: ModelEntry, context: int | None) -> int | None:
        """Misura reale di un caricamento precedente dello stesso modello."""
        if self.store is None:
            return None
        try:
            rows = self.store.model_observations(model.key)
        except Exception:
            return None
        if not rows:
            return None
        # Si preferisce l'osservazione con il contesto piu' vicino a quello
        # richiesto: la cache KV domina il totale e confrontare caricamenti con
        # contesti molto diversi darebbe un numero fuorviante.
        if context:
            rows = sorted(rows, key=lambda r: abs((r.get("context_length") or 0) - context))
        return int(rows[0]["observed_bytes"])
