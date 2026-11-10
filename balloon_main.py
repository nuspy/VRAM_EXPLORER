"""Processo che detiene la riserva di VRAM.

Vive separato dal servizio per due motivi: il contesto CUDA appartiene al
processo che lo crea, e tenere il runtime CUDA fuori dal servizio principale lo
mantiene leggero e rapido da riavviare.

Comunica con il padre stampando una riga JSON con lo stato appena la riserva e'
in piedi, poi resta in attesa finche' non viene terminato.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from vramexp.control.balloon import VramBalloon

_stop = threading.Event()


def _handle_signal(signum: int, frame: object) -> None:
    _stop.set()


def main() -> int:
    parser = argparse.ArgumentParser(description="Detiene una riserva di VRAM")
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--device", type=int, default=0)
    ns = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    balloon = VramBalloon(device=ns.device, chunk_bytes=ns.chunk)
    if not balloon.available:
        print(json.dumps({"allocated_bytes": 0, "error": balloon.error}), flush=True)
        return 1

    state = balloon.resize(ns.bytes)
    print(json.dumps(state.to_dict()), flush=True)

    try:
        # Attesa indefinita: la riserva esiste finche' esiste questo processo.
        while not _stop.wait(1.0):
            pass
    finally:
        balloon.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
