"""Riconoscimento dei processi llama.cpp (`llama-server`, `llama-cli`, `main`).

E' il caso in cui la command line da' quasi tutto: `--model <percorso.gguf>`
identifica esattamente il file, da cui si leggono architettura, quantizzazione e
parametri per la cache KV. I flag `-ngl`, `-c`, `--n-gpu-layers` dicono quanto
del modello e' stato effettivamente messo in VRAM, che e' il dato piu' rilevante
per un explorer della VRAM.

Il server llama.cpp avviato da LM Studio richiede una API key (visibile in
cmdline con `--api-key`): la si riusa per interrogare l'endpoint e confermare il
modello effettivamente caricato.
"""

from __future__ import annotations

import os
import re
import shlex

from ..core.models import Confidence, ContentInfo, ProcInfo
from .base import get_json
from .gguf_header import read_gguf

EXE_NAMES = {"llama-server.exe", "llama-cli.exe", "llama-server", "llama-cli",
             "main.exe", "server.exe", "llama-bench.exe", "llama-embedding.exe"}

_ARG_RE = {
    "model": re.compile(r"(?:^|\s)(?:-m|--model)\s+(?P<v>\"[^\"]+\"|\S+)"),
    "ngl": re.compile(r"(?:^|\s)(?:-ngl|--n-gpu-layers|--gpu-layers)\s+(?P<v>\d+)"),
    "ctx": re.compile(r"(?:^|\s)(?:-c|--ctx-size)\s+(?P<v>\d+)"),
    "port": re.compile(r"(?:^|\s)--port\s+(?P<v>\d+)"),
    "host": re.compile(r"(?:^|\s)--host\s+(?P<v>\S+)"),
    "api_key": re.compile(r"(?:^|\s)--api-key\s+(?P<v>\S+)"),
    "mmproj": re.compile(r"(?:^|\s)--mmproj\s+(?P<v>\"[^\"]+\"|\S+)"),
    "split": re.compile(r"(?:^|\s)--split-mode\s+(?P<v>\S+)"),
    "cache_k": re.compile(r"(?:^|\s)(?:-ctk|--cache-type-k)\s+(?P<v>\S+)"),
    "cache_v": re.compile(r"(?:^|\s)(?:-ctv|--cache-type-v)\s+(?P<v>\S+)"),
}


def _arg(cmdline: str, key: str) -> str | None:
    m = _ARG_RE[key].search(cmdline)
    if not m:
        return None
    return m.group("v").strip('"')


class LlamaCppResolver:
    name = "llama.cpp"

    def matches(self, info: ProcInfo) -> bool:
        if info.name.lower() in EXE_NAMES:
            return True
        return "llama-server" in info.cmdline.lower() or "--n-gpu-layers" in info.cmdline

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        cmd = info.cmdline
        model_path = _arg(cmd, "model")
        if not model_path:
            # Alcune build accettano il modello come argomento posizionale.
            for tok in reversed(_safe_split(cmd)):
                if tok.lower().endswith(".gguf"):
                    model_path = tok
                    break
        if not model_path:
            return None

        ctx = _arg(cmd, "ctx")
        ngl = _arg(cmd, "ngl")
        port = _arg(cmd, "port")
        host = _arg(cmd, "host") or "127.0.0.1"
        api_key = _arg(cmd, "api_key")

        content = ContentInfo(
            kind="llm",
            backend="llama.cpp",
            source=self.name,
            confidence=Confidence.PARSED,
            file_path=model_path,
            model_name=os.path.splitext(os.path.basename(model_path))[0],
        )
        if port:
            content.endpoint = f"http://{host}:{port}"

        gguf = read_gguf(model_path)
        if gguf is not None:
            content.file_size_bytes = gguf.file_size
            content.quantization = gguf.quantization
            content.parameters = gguf.parameters
            content.context_length = int(ctx) if ctx else gguf.context_length
            content.extra.update({
                "architecture": gguf.architecture,
                "n_layers": gguf.n_layers,
                "n_heads": gguf.n_heads,
                "n_kv_heads": gguf.n_kv_heads,
                "head_dim": gguf.head_dim,
                "embedding_length": gguf.embedding_length,
                "trained_context": gguf.context_length,
                "vocab_size": gguf.vocab_size,
            })
            if gguf.n_layers and ngl:
                offloaded = min(int(ngl), gguf.n_layers)
                content.extra["gpu_layers"] = f"{offloaded}/{gguf.n_layers}"
                if offloaded < gguf.n_layers:
                    # Caso importante: parte del modello resta in RAM, quindi la
                    # VRAM occupata sara' minore del file e le prestazioni
                    # peggiori. Va detto, non dedotto dall'utente.
                    content.extra["partial_offload"] = True

        if _arg(cmd, "mmproj"):
            content.kind = "vlm"
            content.extra["multimodal"] = True
        for k in ("cache_k", "cache_v"):
            v = _arg(cmd, k)
            if v:
                content.extra[k] = v

        # Conferma dal server: se risponde, il modello elencato e' quello
        # realmente caricato, non solo quello richiesto sulla riga di comando.
        if content.endpoint:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            data = get_json(f"{content.endpoint}/v1/models", headers=headers)
            if isinstance(data, dict) and data.get("data"):
                served = data["data"][0].get("id")
                if served:
                    content.extra["served_id"] = served
                    content.confidence = Confidence.CONFIRMED

        content.label = content.model_name or "modello llama.cpp"
        content.detail = " · ".join(x for x in (
            content.quantization,
            content.parameters,
            f"ctx {int(content.context_length):,}".replace(",", " ") if content.context_length else None,
            content.extra.get("gpu_layers") and f"layer GPU {content.extra['gpu_layers']}",
        ) if x)
        return content


def _safe_split(cmd: str) -> list[str]:
    try:
        return shlex.split(cmd, posix=False)
    except ValueError:
        return cmd.split()
