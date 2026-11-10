"""Protocollo fra il servizio e l'agent iniettato nei processi Python.

Trasporto: named pipe di Windows, uno per processo
(`\\\\.\\pipe\\vramexp-agent-<pid>`). Rispetto a un socket TCP evita di occupare
porte, non compare nelle scansioni di rete e l'accesso e' regolato dalle ACL del
kernel invece che da un token applicativo.

Messaggi: JSON su una riga, richiesta e risposta. Il volume e' minimo (qualche
chiamata al minuto) e la leggibilita' in fase di diagnosi vale piu' della
compattezza di un formato binario.
"""

from __future__ import annotations

import json
from typing import Any

PIPE_PREFIX = r"\\.\pipe\vramexp-agent-"
MAX_MESSAGE = 8 * 1024 * 1024      # gli snapshot con stack trace possono essere grandi
PROTOCOL_VERSION = 1

# Comandi supportati dall'agent.
CMD_PING = "ping"
CMD_STATS = "stats"                # memory_stats + frammentazione
CMD_SNAPSHOT = "snapshot"          # elenco dei blocchi allocati
CMD_EMPTY_CACHE = "empty_cache"
CMD_RECORD_START = "record_start"  # attiva la storia con stack trace
CMD_RECORD_STOP = "record_stop"
CMD_MODULES = "modules"            # moduli e tensori per modulo


def pipe_name(pid: int) -> str:
    return f"{PIPE_PREFIX}{pid}"


def encode(obj: Any) -> bytes:
    data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    if len(data) > MAX_MESSAGE:
        # Meglio una risposta troncata ed esplicita che una pipe che si blocca.
        data = json.dumps({
            "ok": False,
            "error": f"risposta troppo grande ({len(data)} byte): ridurre il dettaglio richiesto",
        }).encode("utf-8")
    return data + b"\n"


def decode(data: bytes) -> Any:
    return json.loads(data.decode("utf-8", errors="replace"))


def request(command: str, **params: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "cmd": command, "params": params}


def ok(**fields: Any) -> dict[str, Any]:
    return {"ok": True, **fields}


def err(message: str, **fields: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **fields}
