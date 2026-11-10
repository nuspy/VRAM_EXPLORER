"""Avvio della GUI di VRAM Explorer.

Da eseguire con `python vramexp_gui.py`, oppure `pythonw vramexp_gui.py` per
avviarla senza finestra di console.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
