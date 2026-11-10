"""Interfaccia comune dei resolver di contenuto.

Un resolver risponde alla domanda "cosa sta tenendo in VRAM questo processo?".
Ognuno conosce un solo backend e fallisce in silenzio quando il processo non e'
di sua competenza: la catena in `registry.py` li prova in ordine di affidabilita'
e si ferma al primo che risponde con confidenza sufficiente.

Il campo `confidence` e' parte del contratto, non un dettaglio: la GUI mostra un
badge diverso per un dato letto dall'API del backend e per una supposizione
basata sul nome dell'eseguibile, e l'utente deve poter distinguere i due casi
prima di decidere di terminare un processo.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..core.models import ContentInfo, ProcInfo

# Le API locali dei backend rispondono in millisecondi quando ci sono; quando non
# ci sono, un timeout lungo bloccherebbe il ciclo di campionamento.
HTTP_TIMEOUT = 1.5


class Resolver(Protocol):
    name: str

    def matches(self, info: ProcInfo) -> bool:
        """True se questo resolver ha senso per il processo dato."""
        ...

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        """Restituisce il contenuto, o None se non riesce a determinarlo."""
        ...


_client: httpx.Client | None = None


def http() -> httpx.Client:
    """Client HTTP condiviso, con connessioni riusate.

    Creare un client per richiesta significherebbe un handshake TCP nuovo per
    ogni interrogazione, ripetuto per ogni backend a ogni ciclo di refresh.
    """
    global _client
    if _client is None:
        _client = httpx.Client(timeout=HTTP_TIMEOUT,
                               limits=httpx.Limits(max_connections=8,
                                                   max_keepalive_connections=4))
    return _client


def close_http() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_json(url: str, headers: dict[str, str] | None = None) -> object | None:
    """GET che restituisce None invece di sollevare.

    Un backend spento e' la normalita', non un errore: interroghiamo endpoint
    che nella maggior parte dei casi non risponderanno.
    """
    try:
        r = http().get(url, headers=headers)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def post_json(url: str, payload: object | None = None,
              headers: dict[str, str] | None = None) -> tuple[bool, str]:
    """POST per le azioni di unload. Restituisce (successo, messaggio)."""
    try:
        r = http().post(url, json=payload, headers=headers)
        if 200 <= r.status_code < 300:
            return True, r.text[:400]
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
