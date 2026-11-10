"""Consente `python -m vramexp <comando>` senza il warning di runpy che si
otterrebbe invocando direttamente `vramexp.cli.main`."""

import sys

from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
