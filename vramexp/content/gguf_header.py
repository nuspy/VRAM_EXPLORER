"""Lettura dei metadati GGUF senza caricare il modello.

Un GGUF da 20 GB comincia con un blocco di metadati chiave-valore che descrive
l'architettura del modello: numero di layer, teste di attenzione, teste KV
(fondamentali per il calcolo della cache KV), dimensione dell'embedding,
lunghezza di contesto per cui e' stato addestrato. Leggerlo costa qualche
centinaio di kilobyte di I/O.

Serve a due cose: etichettare correttamente cosa occupa la VRAM, e alimentare il
planner "ci sta un 70B con 32k di contesto?" con numeri veri invece che stimati.

Formato: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# Tipi dei valori nei metadati GGUF
(GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16, GGUF_UINT32, GGUF_INT32,
 GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64,
 GGUF_FLOAT64) = range(13)

_SCALAR = {
    GGUF_UINT8: ("<B", 1), GGUF_INT8: ("<b", 1),
    GGUF_UINT16: ("<H", 2), GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4), GGUF_INT32: ("<i", 4),
    GGUF_FLOAT32: ("<f", 4), GGUF_BOOL: ("<?", 1),
    GGUF_UINT64: ("<Q", 8), GGUF_INT64: ("<q", 8),
    GGUF_FLOAT64: ("<d", 8),
}

# Quantizzazione dal nome file, quando i metadati non la riportano.
_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z0-9_]+|Q\d+_[KM0-9_]*[A-Z]*|BF16|F16|F32|FP16|FP8|MXFP4)",
    re.IGNORECASE)
_PARAMS_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BbMm])(?![A-Za-z])")


@dataclass(slots=True)
class GgufInfo:
    path: str
    file_size: int = 0
    architecture: str | None = None
    name: str | None = None
    quantization: str | None = None
    parameters: str | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    embedding_length: int | None = None
    context_length: int | None = None
    rope_freq_base: float | None = None
    vocab_size: int | None = None
    key_length: int | None = None      # dimensione per testa, se dichiarata
    value_length: int | None = None
    kv: dict[str, Any] = field(default_factory=dict)

    @property
    def head_dim(self) -> int | None:
        """Dimensione di una testa di attenzione.

        Va preferito `attention.key_length` quando presente: diverse
        architetture recenti (Gemma 3/4, alcuni Qwen) usano teste piu' larghe di
        `embedding_length / head_count`, e derivarlo per divisione porterebbe a
        sottostimare la cache KV anche del 50%.
        """
        if self.key_length:
            return self.key_length
        if self.embedding_length and self.n_heads:
            return self.embedding_length // self.n_heads
        return None


def _read(f: BinaryIO, fmt: str, size: int) -> Any:
    data = f.read(size)
    if len(data) < size:
        raise EOFError("GGUF troncato")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    n = _read(f, "<Q", 8)
    if n > 16 * 1024 * 1024:
        raise ValueError("stringa GGUF implausibile")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int, depth: int = 0) -> Any:
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return _read(f, fmt, size)
    if vtype == GGUF_STRING:
        return _read_string(f)
    if vtype == GGUF_ARRAY:
        if depth > 2:
            raise ValueError("array GGUF annidato troppo in profondita'")
        elem_type = _read(f, "<I", 4)
        count = _read(f, "<Q", 8)
        # I vocabolari hanno centinaia di migliaia di elementi e non servono a
        # nulla qui: si salta il contenuto conservando solo la lunghezza.
        if count > 1024:
            _skip_array(f, elem_type, count)
            return {"__array_len__": count, "__elem_type__": elem_type}
        return [_read_value(f, elem_type, depth + 1) for _ in range(count)]
    raise ValueError(f"tipo GGUF sconosciuto: {vtype}")


def _skip_array(f: BinaryIO, elem_type: int, count: int) -> None:
    if elem_type in _SCALAR:
        f.seek(_SCALAR[elem_type][1] * count, 1)
    elif elem_type == GGUF_STRING:
        for _ in range(count):
            n = _read(f, "<Q", 8)
            f.seek(n, 1)
    else:
        raise ValueError("array di tipo non saltabile")


def read_gguf(path: str | Path, max_kv: int = 2000) -> GgufInfo | None:
    """Legge l'intestazione di un GGUF. `None` se il file non e' un GGUF valido.

    Non solleva su file corrotti o troncati: il tool deve continuare a
    funzionare anche davanti a un download interrotto a meta'.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return None

    info = GgufInfo(path=str(p), file_size=size)
    try:
        with p.open("rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            _version = _read(f, "<I", 4)
            _n_tensors = _read(f, "<Q", 8)
            n_kv = _read(f, "<Q", 8)

            for _ in range(min(n_kv, max_kv)):
                key = _read_string(f)
                vtype = _read(f, "<I", 4)
                try:
                    info.kv[key] = _read_value(f, vtype)
                except (ValueError, EOFError):
                    break   # metadati parziali sono comunque meglio di niente
    except (OSError, EOFError, struct.error, ValueError):
        pass

    _populate(info)
    return info


def _populate(info: GgufInfo) -> None:
    kv = info.kv
    arch = kv.get("general.architecture")
    info.architecture = arch if isinstance(arch, str) else None
    name = kv.get("general.name")
    info.name = name if isinstance(name, str) else None

    if arch:
        def g(suffix: str) -> Any:
            return kv.get(f"{arch}.{suffix}")
        info.n_layers = _as_int(g("block_count"))
        info.n_heads = _as_int(g("attention.head_count"))
        info.n_kv_heads = _as_int(g("attention.head_count_kv"))
        info.embedding_length = _as_int(g("embedding_length"))
        info.context_length = _as_int(g("context_length"))
        info.key_length = _as_int(g("attention.key_length"))
        info.value_length = _as_int(g("attention.value_length"))
        rope = g("rope.freq_base")
        info.rope_freq_base = float(rope) if isinstance(rope, (int, float)) else None

    # Se il modello non usa GQA, n_kv_heads coincide con n_heads: assumerlo evita
    # di sovrastimare enormemente la cache KV nel planner.
    if info.n_kv_heads is None and info.n_heads is not None:
        info.n_kv_heads = info.n_heads

    vocab = kv.get("tokenizer.ggml.tokens")
    if isinstance(vocab, dict) and "__array_len__" in vocab:
        info.vocab_size = int(vocab["__array_len__"])

    ftype = kv.get("general.file_type")
    info.quantization = _quant_from_name(Path(info.path).name) or _ftype_name(ftype)
    info.parameters = _params_from_name(Path(info.path).name) or _params_from_kv(kv)


def _as_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _quant_from_name(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _params_from_name(filename: str) -> str | None:
    for m in _PARAMS_RE.finditer(filename):
        value, unit = m.group(1), m.group(2).upper()
        # Scarta numeri che chiaramente non sono conteggi di parametri
        # (versioni, quantizzazioni gia' catturate altrove).
        if unit == "B" and 0.1 <= float(value) <= 2000:
            return f"{value}B"
        if unit == "M" and 1 <= float(value) <= 999:
            return f"{value}M"
    return None


def _params_from_kv(kv: dict[str, Any]) -> str | None:
    n = kv.get("general.parameter_count")
    if isinstance(n, (int, float)) and n > 0:
        if n >= 1e9:
            return f"{n / 1e9:.0f}B"
        return f"{n / 1e6:.0f}M"
    return None


_FTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


def _ftype_name(ftype: Any) -> str | None:
    if isinstance(ftype, int):
        return _FTYPE_NAMES.get(ftype)
    return None
