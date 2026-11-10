"""Identita' dei processi: nome, cmdline, utente, editore, icona.

Queste informazioni sono molto piu' costose da leggere dei contatori di memoria
(aprono handle, leggono il PEB, verificano firme digitali su disco), ma cambiano
solo alla nascita del processo. Vengono quindi raccolte **una volta per PID** e
tenute in cache, mentre i campioni di memoria scorrono a 1 Hz.

La cmdline e' anche la fonte principale per capire *cosa* c'e' in VRAM: e' da li'
che si ricava, ad esempio, il percorso del GGUF passato a llama-server.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from dataclasses import dataclass

import psutil

from ..core.models import ProcInfo

# Processi la cui terminazione destabilizza la sessione o l'intero sistema.
# Restano visibili e analizzabili: si nega solo l'azione distruttiva.
PROTECTED_NAMES = {
    "dwm.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe",
    "lsass.exe", "smss.exe", "system", "registry", "memory compression",
    "explorer.exe", "svchost.exe", "fontdrvhost.exe", "audiodg.exe",
    "sihost.exe", "ctfmon.exe", "logonui.exe", "textinputhost.exe",
}

SYSTEM_DIRS = (
    os.environ.get("SystemRoot", r"C:\Windows").lower(),
)


@dataclass(slots=True)
class _CacheEntry:
    info: ProcInfo
    create_time: float
    fetched_at: float


class ProcSource:
    """Cache di identita' processo, invalidata su riuso del PID.

    Windows ricicla i PID: la chiave della cache e' quindi (pid, create_time),
    altrimenti un processo nuovo erediterebbe l'identita' di uno morto.
    """

    def __init__(self, want_icons: bool = True) -> None:
        self._cache: dict[int, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._want_icons = want_icons
        self._icon_cache: dict[str, str | None] = {}

    def get(self, pid: int) -> ProcInfo | None:
        with self._lock:
            entry = self._cache.get(pid)
        if entry is not None:
            try:
                if psutil.Process(pid).create_time() == entry.create_time:
                    return entry.info
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            with self._lock:
                self._cache.pop(pid, None)
        return self._fetch(pid)

    def _fetch(self, pid: int) -> ProcInfo | None:
        try:
            p = psutil.Process(pid)
            with p.oneshot():
                name = p.name()
                create_time = p.create_time()
                try:
                    exe = p.exe()
                except (psutil.AccessDenied, OSError):
                    exe = None
                try:
                    cmdline = " ".join(p.cmdline())
                except (psutil.AccessDenied, OSError):
                    cmdline = ""
                try:
                    username = p.username()
                except (psutil.AccessDenied, OSError):
                    username = None
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            # Processi di sistema protetti: si conserva comunque il PID, cosi'
            # la GUI puo' mostrarne il consumo invece di farlo sparire.
            return ProcInfo(pid=pid, name=f"pid {pid}", cmdline="",
                            username=None, publisher="<accesso negato>")

        info = ProcInfo(
            pid=pid, name=name, exe=exe, cmdline=cmdline,
            username=username, create_time=create_time,
            publisher=_publisher(exe),
            icon_png_b64=self._icon(exe) if self._want_icons else None,
        )
        with self._lock:
            self._cache[pid] = _CacheEntry(info=info, create_time=create_time,
                                           fetched_at=time.time())
        return info

    def _icon(self, exe: str | None) -> str | None:
        if not exe:
            return None
        if exe in self._icon_cache:
            return self._icon_cache[exe]
        png = _extract_icon_png(exe)
        self._icon_cache[exe] = png
        return png

    def prune(self, live_pids: set[int]) -> None:
        with self._lock:
            for pid in list(self._cache):
                if pid not in live_pids:
                    del self._cache[pid]


def is_protected(info: ProcInfo) -> bool:
    """True se il processo non deve essere terminato dall'applicazione."""
    if info.name.lower() in PROTECTED_NAMES:
        return True
    if info.exe:
        low = info.exe.lower()
        if any(low.startswith(d) for d in SYSTEM_DIRS if d):
            # Gli eseguibili sotto %SystemRoot% sono di sistema, con l'eccezione
            # nota di quelli sotto WindowsApps (app dello Store, terminabili).
            return "windowsapps" not in low
    return False


def is_system_process(info: ProcInfo) -> bool:
    """Distingue infrastruttura Windows da applicazioni dell'utente, per il
    filtro "mostra processi di sistema" della GUI."""
    if info.name.lower() in PROTECTED_NAMES:
        return True
    if info.exe:
        low = info.exe.lower()
        return any(low.startswith(d) for d in SYSTEM_DIRS if d) and "windowsapps" not in low
    return False


def _publisher(exe: str | None) -> str | None:
    """Editore dichiarato nelle risorse di versione del file.

    Si legge la risorsa VERSIONINFO (CompanyName), non la firma Authenticode:
    verificare la catena di certificati per ogni processo a ogni avvio sarebbe
    lento e qui serve solo un'etichetta per l'utente, non una prova di fiducia.
    """
    if not exe or not os.path.exists(exe):
        return None
    try:
        import win32api
        info = win32api.GetFileVersionInfo(exe, "\\VarFileInfo\\Translation")
        if not info:
            return None
        lang, codepage = info[0]
        key = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\CompanyName"
        return win32api.GetFileVersionInfo(exe, key) or None
    except Exception:
        return None


def _extract_icon_png(exe: str, size: int = 32) -> str | None:
    """Icona dell'eseguibile come PNG in base64, per la GUI e l'API.

    Restituisce None se l'icona non e' estraibile: e' un dettaglio estetico e non
    deve mai bloccare la raccolta dati.
    """
    if not exe or not os.path.exists(exe):
        return None
    try:
        import win32con
        import win32gui
        import win32ui
        from PIL import Image

        large, small = win32gui.ExtractIconEx(exe, 0)
        handles = large + small
        if not handles:
            return None
        hicon = (large or small)[0]
        try:
            hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(hdc, size, size)
            mem_dc = hdc.CreateCompatibleDC()
            mem_dc.SelectObject(bmp)
            mem_dc.DrawIcon((0, 0), hicon)

            bits = bmp.GetBitmapBits(True)
            img = Image.frombuffer("RGBA", (size, size), bits, "raw", "BGRA", 0, 1)
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            mem_dc.DeleteDC()
            hdc.DeleteDC()
            win32gui.DeleteObject(bmp.GetHandle())
            return base64.b64encode(buf.getvalue()).decode("ascii")
        finally:
            for h in handles:
                try:
                    win32gui.DestroyIcon(h)
                except Exception:
                    pass
    except Exception:
        return None
