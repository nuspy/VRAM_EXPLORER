"""Punto di ingresso del servizio VRAM Explorer.

Avvia il ciclo di campionamento e il server API, e resta in esecuzione. Progettato
per girare sotto NSSM come servizio Windows con riavvio automatico: per questo
qualunque errore fatale termina il processo con codice diverso da zero invece di
tentare di sopravvivere in uno stato degradato — il supervisore lo riavvia
pulito, che e' piu' affidabile di un processo vivo a meta'.

Il log va su file con rotazione, perche' sotto servizio non c'e' una console a
cui guardare.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import signal
import sys
import threading

import uvicorn

from vramexp.api.server import create_app
from vramexp.core.config import Config, get_or_create_token, log_dir
from vramexp.service import VramService

_shutdown = threading.Event()


def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir() / "service.log", maxBytes=8 * 1024 * 1024, backupCount=5,
        encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # uvicorn e' loquace sui log di accesso e qui non servono: ogni richiesta
    # della GUI produrrebbe una riga.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="Servizio VRAM Explorer")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-api", action="store_true",
                        help="solo campionamento e storico, senza server HTTP")
    ns = parser.parse_args()

    setup_logging(ns.verbose)
    log = logging.getLogger("vramexp.main")

    cfg = Config.load()
    if ns.host:
        cfg.api_host = ns.host
    if ns.port:
        cfg.api_port = ns.port

    if cfg.api_host not in ("127.0.0.1", "localhost", "::1"):
        # Il servizio espone terminazione di processi e modifica delle
        # impostazioni del driver: non deve raggiungere la rete, mai.
        log.error("host %s non consentito: l'API resta vincolata al loopback",
                  cfg.api_host)
        cfg.api_host = "127.0.0.1"

    service = VramService(cfg)
    health = service.sampler.health()
    log.info("collector: NVML=%s PDH=%s GPU=%s",
             health["nvml_available"], health["pdh_available"], health["gpu_count"])
    if not health["pdh_available"]:
        log.error("contatori PDH non disponibili: la memoria per processo non sara' "
                  "leggibile. %s", health.get("pdh_error"))

    service.start()

    def handle_signal(signum: int, frame: object) -> None:
        log.info("segnale %s ricevuto: arresto in corso", signum)
        _shutdown.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if ns.no_api:
        try:
            while not _shutdown.wait(1.0):
                pass
        finally:
            service.stop()
        return 0

    token = get_or_create_token() if cfg.api_require_token else None
    log.info("API su http://%s:%s (token in %%LOCALAPPDATA%%\\VramExplorer)",
             cfg.api_host, cfg.api_port)
    if token:
        log.debug("token attivo (lunghezza %d)", len(token))

    app = create_app(service, cfg)
    config = uvicorn.Config(app, host=cfg.api_host, port=cfg.api_port,
                            log_level="debug" if ns.verbose else "info",
                            access_log=False)
    server = uvicorn.Server(config)

    def stop_server() -> None:
        _shutdown.wait()
        server.should_exit = True

    threading.Thread(target=stop_server, daemon=True).start()

    try:
        server.run()
    except Exception:
        log.exception("il server API e' terminato con errore")
        service.stop()
        return 1
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
