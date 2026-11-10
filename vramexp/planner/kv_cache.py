"""Stima del fabbisogno di VRAM per un modello linguistico.

Tre voci compongono il totale:

1. **Pesi** — la dimensione del file quantizzato e' un'ottima approssimazione:
   i tensori vengono caricati cosi' come sono. Se solo una parte dei layer va in
   GPU, si scala di conseguenza.

2. **Cache KV** — cresce linearmente con il contesto ed e' la voce che sorprende:
   con contesti da 128k puo' superare i pesi stessi. Formula:

       2 (chiavi e valori) × n_layers × n_kv_heads × head_dim × contesto × byte_per_elemento

   Il fattore decisivo e' `n_kv_heads`, non `n_heads`: con GQA (Grouped Query
   Attention) le teste KV sono molte meno di quelle di query, e usare `n_heads`
   sovrastimerebbe di un ordine di grandezza. Sul Qwen3.6-35B misurato:
   40 layer, 2 teste KV, head_dim 256 — con `n_heads` verrebbe fuori una cifra
   quattro volte piu' grande.

3. **Overhead** — contesto CUDA, buffer di calcolo, frammentazione
   dell'allocatore. Si stima in modo prudente perche' sottostimare produce un
   "ci sta" che poi fallisce, che e' l'esito peggiore per chi usa lo strumento.

Le stime sono dichiarate come tali. Quando lo storico contiene una misura reale
dello stesso modello, `whatif.py` la preferisce al calcolo.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..content.gguf_header import GgufInfo

# Byte per elemento nella cache KV, per tipo di quantizzazione della cache.
KV_TYPE_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625,     # 1 byte piu' la scala per blocco
    "q5_1": 0.75, "q5_0": 0.6875,
    "q4_1": 0.5625, "q4_0": 0.5625,
}

# Il contesto CUDA da solo occupa VRAM prima di qualunque allocazione: misurato
# a ~2,7 GB sulla RTX 5090 con driver 610. Si usa un valore prudente.
CUDA_CONTEXT_BYTES = 700 * 1024 * 1024

# Buffer di calcolo di llama.cpp: dipendono da batch e dimensione del modello.
COMPUTE_BUFFER_MIN = 256 * 1024 * 1024


@dataclass
class VramEstimate:
    weights_bytes: int = 0
    kv_cache_bytes: int = 0
    overhead_bytes: int = 0
    total_bytes: int = 0
    context_length: int = 0
    is_estimate: bool = True
    confidence_note: str = ""
    breakdown: dict[str, int] | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "weights_bytes": self.weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "overhead_bytes": self.overhead_bytes,
            "total_bytes": self.total_bytes,
            "context_length": self.context_length,
            "is_estimate": self.is_estimate,
            "confidence_note": self.confidence_note,
            "breakdown": self.breakdown or {},
            "warnings": self.warnings or [],
        }


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, context: int,
                   kv_type: str = "f16", batch: int = 1) -> int:
    """Dimensione della cache KV. Vedi la formula nel docstring del modulo."""
    per_element = KV_TYPE_BYTES.get(kv_type.lower(), 2.0)
    return int(2 * n_layers * n_kv_heads * head_dim * context * per_element * batch)


def estimate_gguf(gguf: GgufInfo, context: int | None = None,
                  gpu_layers: int | None = None, kv_type: str = "f16",
                  batch: int = 1) -> VramEstimate:
    """Stima il fabbisogno per un GGUF con i parametri di caricamento indicati."""
    warnings: list[str] = []
    ctx = context or gguf.context_length or 4096

    weights = gguf.file_size
    if gpu_layers is not None and gguf.n_layers:
        if gpu_layers < gguf.n_layers:
            fraction = max(0.0, min(1.0, gpu_layers / gguf.n_layers))
            weights = int(weights * fraction)
            warnings.append(
                f"solo {gpu_layers} layer su {gguf.n_layers} in GPU: i restanti "
                "girano sulla CPU e rallentano molto la generazione")

    kv = 0
    if gguf.n_layers and gguf.n_kv_heads and gguf.head_dim:
        kv = kv_cache_bytes(gguf.n_layers, gguf.n_kv_heads, gguf.head_dim, ctx,
                            kv_type, batch)
        if gguf.n_heads and gguf.n_kv_heads < gguf.n_heads:
            ratio = gguf.n_heads / gguf.n_kv_heads
            warnings.append(
                f"il modello usa GQA ({gguf.n_kv_heads} teste KV su {gguf.n_heads} "
                f"di query): la cache e' {ratio:.0f} volte piu' piccola di quanto "
                "sarebbe con attenzione piena")
    else:
        warnings.append("metadati insufficienti per calcolare la cache KV: "
                        "stima basata sui soli pesi")

    # I buffer di calcolo crescono con il modello; il 3% dei pesi e' una regola
    # empirica che si e' rivelata prudente sui modelli di taglia comune.
    compute = max(COMPUTE_BUFFER_MIN, int(weights * 0.03))
    overhead = CUDA_CONTEXT_BYTES + compute

    total = weights + kv + overhead
    return VramEstimate(
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=overhead,
        total_bytes=total,
        context_length=ctx,
        is_estimate=True,
        confidence_note=("stima calcolata dai metadati del file; "
                         "lo scarto tipico e' entro il 10%"),
        breakdown={
            "pesi": weights,
            "cache_kv": kv,
            "contesto_cuda": CUDA_CONTEXT_BYTES,
            "buffer_calcolo": compute,
        },
        warnings=warnings,
    )


def max_context_for_budget(gguf: GgufInfo, budget_bytes: int,
                           kv_type: str = "f16") -> int:
    """Contesto massimo che entra in un budget dato.

    Risponde alla domanda pratica "quanto contesto posso permettermi con la VRAM
    che ho", che e' l'inverso della stima e spesso piu' utile.
    """
    if not (gguf.n_layers and gguf.n_kv_heads and gguf.head_dim):
        return 0
    fixed = gguf.file_size + CUDA_CONTEXT_BYTES + max(
        COMPUTE_BUFFER_MIN, int(gguf.file_size * 0.03))
    available = budget_bytes - fixed
    if available <= 0:
        return 0
    per_token = (2 * gguf.n_layers * gguf.n_kv_heads * gguf.head_dim
                 * KV_TYPE_BYTES.get(kv_type.lower(), 2.0))
    if per_token <= 0:
        return 0
    ctx = int(available / per_token)
    # Arrotondato a multipli di 1024 e limitato al contesto per cui il modello e'
    # stato addestrato: proporre 300k a un modello da 128k non aiuterebbe.
    ctx = (ctx // 1024) * 1024
    if gguf.context_length:
        ctx = min(ctx, gguf.context_length)
    return max(0, ctx)
