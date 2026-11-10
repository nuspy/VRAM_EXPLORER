"""Conversioni e formattazione delle dimensioni di memoria.

Tutta la codebase lavora internamente in **byte** (int). La conversione a unita'
leggibili avviene solo al confine con l'utente (GUI, CLI, API di presentazione).
"""

from __future__ import annotations

KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

_UNITS = (("TB", 1024**4), ("GB", GIB), ("MB", MIB), ("KB", KIB))


def human_bytes(n: int | float | None, decimals: int = 1) -> str:
    """Formatta un numero di byte in stringa leggibile (base 1024).

    `None` diventa "n/d" invece di sollevare: i collector possono legittimamente
    non avere un valore per una metrica non supportata dall'hardware.
    """
    if n is None:
        return "n/d"
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for suffix, factor in _UNITS:
        if n >= factor:
            return f"{sign}{n / factor:.{decimals}f} {suffix}"
    return f"{sign}{n:.0f} B"


def mib(n: int | float | None) -> float | None:
    return None if n is None else n / MIB


def gib(n: int | float | None) -> float | None:
    return None if n is None else n / GIB


def parse_size(text: str) -> int:
    """Converte "8G", "512MB", "1.5 GiB", "1073741824" in byte.

    Usata da CLI e file di configurazione, dove scrivere i byte a mano sarebbe
    scomodo e fonte di errori.
    """
    s = text.strip().upper().replace("IB", "B").replace(" ", "")
    if not s:
        raise ValueError("dimensione vuota")
    factors = {"TB": 1024**4, "GB": GIB, "MB": MIB, "KB": KIB, "B": 1,
               "T": 1024**4, "G": GIB, "M": MIB, "K": KIB}
    for suffix in ("TB", "GB", "MB", "KB", "B", "T", "G", "M", "K"):
        if s.endswith(suffix):
            number = s[: -len(suffix)]
            if not number:
                raise ValueError(f"dimensione non valida: {text!r}")
            return int(float(number) * factors[suffix])
    return int(float(s))


def pct(part: float | None, whole: float | None) -> float:
    """Percentuale robusta: 0.0 se il denominatore e' assente o nullo."""
    if not whole or part is None:
        return 0.0
    return max(0.0, min(100.0, part / whole * 100.0))
