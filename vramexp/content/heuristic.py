"""Classificazione di riserva, quando nessun backend riconosce il processo.

E' l'ultimo anello della catena e non pretende di sapere *cosa* c'e' in VRAM: si
limita a dire di che genere di consumatore si tratta, il che basta per rendere la
lista leggibile invece di lasciare venti righe "python.exe".

Tutti i risultati escono con confidenza INFERRED, perche' sono supposizioni
basate sul nome dell'eseguibile. La distinzione conta: l'utente non deve
terminare un processo perche' un'euristica ha indovinato male.

Menzione a parte per il compositor: `dwm.exe` risulta spesso fra i primi
consumatori (12,4 GB sulla macchina di sviluppo) e la ragione non e' ovvia — e'
memoria di composizione delle finestre, in larga parte attribuibile ai browser e
alle app accelerate aperte. La spiegazione viene mostrata all'utente perche'
altrimenti la reazione naturale sarebbe tentare di terminarlo.
"""

from __future__ import annotations

from ..core.models import Confidence, ContentInfo, ProcInfo

COMPOSITOR = {"dwm.exe"}

BROWSERS = {
    "chrome.exe": "Google Chrome", "msedge.exe": "Microsoft Edge",
    "firefox.exe": "Mozilla Firefox", "brave.exe": "Brave",
    "opera.exe": "Opera", "vivaldi.exe": "Vivaldi",
    "msedgewebview2.exe": "WebView2 (app incorporata)",
    "chrome_proxy.exe": "Chrome",
}

ELECTRON_HINTS = ("electron", "discord", "slack", "teams", "code.exe",
                  "obsidian", "notion", "spotify", "whatsapp")

CREATIVE = {
    "blender.exe": ("3d", "Blender"),
    "photoshop.exe": ("image", "Adobe Photoshop"),
    "aftereffects.exe": ("video", "Adobe After Effects"),
    "adobe premiere pro.exe": ("video", "Adobe Premiere Pro"),
    "davinciresolve.exe": ("video", "DaVinci Resolve"),
    "obs64.exe": ("video", "OBS Studio"),
    "unrealeditor.exe": ("3d", "Unreal Editor"),
    "unity.exe": ("3d", "Unity"),
    "snap17.exe": ("image", "Ashampoo Snap"),
    "handbrake.exe": ("video", "HandBrake"),
    "ffmpeg.exe": ("video", "FFmpeg"),
}

ML_HINTS = (
    ("torch", "PyTorch"), ("tensorflow", "TensorFlow"), ("onnxruntime", "ONNX Runtime"),
    ("transformers", "Transformers"), ("diffusers", "Diffusers"), ("whisper", "Whisper"),
    ("kokoro", "Kokoro TTS"), ("xtts", "XTTS"), ("vllm", "vLLM"),
    ("stable", "Stable Diffusion"), ("llama", "llama.cpp"), ("exllama", "ExLlama"),
    ("faster_whisper", "faster-whisper"), ("sentence_transformers", "SentenceTransformers"),
)

SHELL = {
    "csrss.exe": "Sottosistema runtime client/server",
    "winlogon.exe": "Accesso di Windows",
    "explorer.exe": "Esplora file", "searchhost.exe": "Ricerca di Windows",
    "startmenuexperiencehost.exe": "Menu Start", "shellexperiencehost.exe": "Shell di Windows",
    "textinputhost.exe": "Input di testo", "applicationframehost.exe": "Contenitore app",
    "windowsterminal.exe": "Windows Terminal", "shellhost.exe": "Shell di Windows",
    "phoneexperiencehost.exe": "Collegamento al telefono",
    "crossdeviceresume.exe": "Continua su dispositivi",
}


class HeuristicResolver:
    name = "euristica"

    def matches(self, info: ProcInfo) -> bool:
        return True   # e' il fallback: accetta tutto

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        name = info.name.lower()
        cmd = info.cmdline.lower()

        if name in COMPOSITOR:
            return ContentInfo(
                kind="compositor", label="Compositor di Windows",
                detail=("superfici delle finestre di tutte le app accelerate; "
                        "cresce con browser, video e finestre aperte"),
                confidence=Confidence.INFERRED, source=self.name,
                extra={"advice": (
                    "Non terminabile. Per ridurne il consumo: chiudere schede del "
                    "browser, disattivare l'accelerazione hardware nelle app che non "
                    "ne hanno bisogno, ridurre il numero di finestre aperte.")},
            )

        if name in BROWSERS:
            return ContentInfo(
                kind="browser", label=BROWSERS[name],
                detail="accelerazione hardware: texture, video decodificati, canvas WebGL",
                confidence=Confidence.INFERRED, source=self.name,
            )

        if name in SHELL:
            return ContentInfo(kind="system", label=SHELL[name],
                               detail="componente dell'interfaccia di Windows",
                               confidence=Confidence.INFERRED, source=self.name)

        if name in CREATIVE:
            kind, label = CREATIVE[name]
            return ContentInfo(kind=kind, label=label,
                               detail="applicazione con accelerazione GPU",
                               confidence=Confidence.INFERRED, source=self.name)

        if name in ("python.exe", "pythonw.exe", "python3.exe", "py.exe"):
            return self._python(info, cmd)

        if any(h in name for h in ELECTRON_HINTS):
            return ContentInfo(kind="app", label=info.name.removesuffix(".exe").title(),
                               detail="applicazione Electron con accelerazione GPU",
                               confidence=Confidence.INFERRED, source=self.name)

        return ContentInfo(kind="unknown", label=info.name or f"pid {info.pid}",
                           detail="contenuto non identificato",
                           confidence=Confidence.UNKNOWN, source=self.name)

    def _python(self, info: ProcInfo, cmd: str) -> ContentInfo:
        """Processi Python: il caso piu' frequente e piu' opaco.

        Senza l'agent di introspezione non si puo' sapere cosa contenga la VRAM,
        ma il nome dello script e le librerie citate nella cmdline restringono
        molto il campo, e il percorso dello script e' comunque l'informazione che
        l'utente cerca per capire "quale mio programma e' questo".
        """
        script = _script_name(info.cmdline)
        framework = next((label for token, label in ML_HINTS if token in cmd), None)

        label = script or "script Python"
        detail_parts = []
        if framework:
            detail_parts.append(framework)
        detail_parts.append("introspezione dettagliata disponibile con l'agent (VRAMEXP_AGENT=1)")

        return ContentInfo(
            kind="ml" if framework else "script",
            label=label,
            detail=" · ".join(detail_parts),
            confidence=Confidence.INFERRED,
            source=self.name,
            file_path=_script_path(info.cmdline),
            extra={"framework": framework, "agent_hint": True},
        )


def _script_path(cmdline: str) -> str | None:
    """Estrae il percorso dello script Python dalla command line.

    Gestisce vari formati:
      - python script.py
      - python.exe "C:\path\to\script.py"
      - python -m modulo
      - "C:\Python\python.exe" script.py
    """
    if not cmdline:
        return None

    # Rimuovi quote esterne e split mantenendo i token quotati
    tokens = []
    current_token = ""
    in_quotes = False
    for char in cmdline:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ' ' and not in_quotes:
            if current_token:
                tokens.append(current_token.strip('"'))
                current_token = ""
        else:
            current_token += char
    if current_token:
        tokens.append(current_token.strip('"'))

    # Cerca il primo token che termina con .py
    for tok in tokens:
        if tok.lower().endswith(".py") and len(tok) > 3:
            # Valida che il path abbia un aspetto ragionevole
            if "\\" in tok or "/" in tok or tok.count(".") == 1:
                return tok

    return None


def _script_name(cmdline: str) -> str | None:
    path = _script_path(cmdline)
    if not path:
        return None
    return path.replace("\\", "/").rsplit("/", 1)[-1]
