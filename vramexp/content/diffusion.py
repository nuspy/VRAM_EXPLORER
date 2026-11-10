"""Backend di diffusione: ComfyUI, Automatic1111/Forge.

Sono i consumatori di VRAM piu' opachi da fuori: girano dentro un `python.exe`
generico, quindi senza interrogare la loro API l'unica cosa visibile sarebbe
"python.exe, 12 GB". L'API invece dice quale checkpoint e' caricato, cosa c'e' in
coda e — per ComfyUI — la ripartizione della memoria vista dal backend stesso.

Il match sul processo avviene per porta: si associa un endpoint a un PID solo se
quel PID e' in ascolto su quella porta, altrimenti su una macchina con piu'
istanze Python si attribuirebbe il contenuto al processo sbagliato.
"""

from __future__ import annotations

import time

import psutil

from ..core.config import Config
from ..core.models import Confidence, ContentInfo, ProcInfo
from .base import get_json

CACHE_TTL = 8.0


def listening_ports(pid: int) -> set[int]:
    """Porte TCP in ascolto di un processo.

    `net_connections` richiede privilegi elevati per i processi di altri utenti;
    per i processi dell'utente corrente (il caso che ci interessa) funziona.
    """
    ports: set[int] = set()
    try:
        for c in psutil.Process(pid).net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr:
                ports.add(c.laddr.port)
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        pass
    return ports


class _CachedProbe:
    """Interroga un insieme di URL e ricorda quali rispondono."""

    def __init__(self, urls: list[str], path: str) -> None:
        self.urls = urls
        self.path = path
        self._cache: dict[str, object] = {}
        self._ts = 0.0

    def probe(self) -> dict[str, object]:
        now = time.time()
        if (now - self._ts) < CACHE_TTL:
            return self._cache
        found: dict[str, object] = {}
        for url in self.urls:
            data = get_json(f"{url}{self.path}")
            if data is not None:
                found[url] = data
        self._cache = found
        self._ts = now
        return found


class ComfyUIResolver:
    name = "comfyui"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._probe = _CachedProbe(self.cfg.comfyui_urls, "/system_stats")

    def matches(self, info: ProcInfo) -> bool:
        low = info.cmdline.lower()
        return "comfyui" in low or "comfy" in (info.exe or "").lower()

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        alive = self._probe.probe()
        endpoint = self._endpoint_for(info, alive)
        if endpoint is None:
            if not self.matches(info):
                return None
            # Riconosciuto dalla cmdline ma API non raggiungibile: si dichiara
            # cio' che si sa, con confidenza inferiore.
            return ContentInfo(kind="diffusion", backend="comfyui", source=self.name,
                               confidence=Confidence.PARSED, label="ComfyUI",
                               detail="API non raggiungibile")

        stats = alive[endpoint]
        content = ContentInfo(
            kind="diffusion", backend="comfyui", source=self.name,
            confidence=Confidence.CONFIRMED, endpoint=endpoint, label="ComfyUI",
        )
        if isinstance(stats, dict):
            devices = stats.get("devices") or []
            if devices and isinstance(devices[0], dict):
                d = devices[0]
                content.extra["torch_vram_total"] = d.get("vram_total")
                content.extra["torch_vram_free"] = d.get("vram_free")
                content.extra["device_name"] = d.get("name")
            sysinfo = stats.get("system") or {}
            content.extra["comfyui_version"] = sysinfo.get("comfyui_version")
            content.extra["python_version"] = str(sysinfo.get("python_version") or "")[:12]

        queue = get_json(f"{endpoint}/queue")
        running = pending = 0
        if isinstance(queue, dict):
            running = len(queue.get("queue_running") or [])
            pending = len(queue.get("queue_pending") or [])
        content.extra["queue_running"] = running
        content.extra["queue_pending"] = pending

        parts = []
        if running:
            parts.append(f"{running} in esecuzione")
        if pending:
            parts.append(f"{pending} in coda")
        if not parts:
            parts.append("inattivo — la VRAM resta occupata dai modelli in cache")
        content.detail = " · ".join(parts)
        return content

    def _endpoint_for(self, info: ProcInfo, alive: dict[str, object]) -> str | None:
        if not alive:
            return None
        ports = listening_ports(info.pid)
        for url in alive:
            try:
                port = int(url.rsplit(":", 1)[1])
            except (ValueError, IndexError):
                continue
            if port in ports:
                return url
        # Un solo backend vivo e un solo processo candidato: attribuzione
        # ragionevole anche senza conferma dalla porta.
        if len(alive) == 1 and self.matches(info):
            return next(iter(alive))
        return None


class A1111Resolver:
    name = "a1111"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._probe = _CachedProbe(self.cfg.a1111_urls, "/sdapi/v1/options")

    def matches(self, info: ProcInfo) -> bool:
        low = info.cmdline.lower()
        return any(k in low for k in ("stable-diffusion-webui", "webui.py", "launch.py", "forge"))

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        alive = self._probe.probe()
        endpoint = None
        ports = listening_ports(info.pid)
        for url in alive:
            try:
                if int(url.rsplit(":", 1)[1]) in ports:
                    endpoint = url
                    break
            except (ValueError, IndexError):
                continue
        if endpoint is None and len(alive) == 1 and self.matches(info):
            endpoint = next(iter(alive))
        if endpoint is None:
            return None

        opts = alive[endpoint]
        content = ContentInfo(
            kind="diffusion", backend="a1111", source=self.name,
            confidence=Confidence.CONFIRMED, endpoint=endpoint,
            label="Stable Diffusion WebUI",
        )
        if isinstance(opts, dict):
            ckpt = opts.get("sd_model_checkpoint")
            if ckpt:
                content.model_name = str(ckpt)
                content.label = str(ckpt)
            content.extra["vae"] = opts.get("sd_vae")

        mem = get_json(f"{endpoint}/sdapi/v1/memory")
        if isinstance(mem, dict):
            cuda = mem.get("cuda") or {}
            system = cuda.get("system") or {}
            if system.get("used"):
                content.extra["torch_used"] = system.get("used")
                content.extra["torch_total"] = system.get("total")
        content.detail = " · ".join(x for x in (
            content.extra.get("vae") and f"VAE {content.extra['vae']}", "diffusione") if x)
        return content


class VllmResolver:
    """vLLM e altri server OpenAI-compatibili avviati a mano."""

    name = "vllm"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def matches(self, info: ProcInfo) -> bool:
        low = info.cmdline.lower()
        return "vllm" in low or "text-generation-inference" in low or "sglang" in low

    def resolve(self, info: ProcInfo) -> ContentInfo | None:
        if not self.matches(info):
            return None
        low = info.cmdline
        content = ContentInfo(kind="llm", backend="vllm", source=self.name,
                              confidence=Confidence.PARSED, label="vLLM")
        # `--model <id>` e' il flag comune a vLLM e SGLang.
        import re
        m = re.search(r"--model[= ]\s*(?P<v>\"[^\"]+\"|\S+)", low)
        if m:
            content.model_name = m.group("v").strip('"')
            content.label = content.model_name
        m = re.search(r"--max-model-len[= ]\s*(\d+)", low)
        if m:
            content.context_length = int(m.group(1))
        m = re.search(r"--gpu-memory-utilization[= ]\s*([\d.]+)", low)
        if m:
            # vLLM preassegna una frazione fissa della VRAM: spiega perche' il
            # processo occupa molto piu' del peso del modello.
            content.extra["gpu_memory_utilization"] = float(m.group(1))
            content.detail = (f"preallocazione {float(m.group(1)) * 100:.0f}% della VRAM "
                              "(comportamento normale di vLLM)")
        for port in _ports_from_cmdline(low) or listening_ports(info.pid):
            data = get_json(f"http://127.0.0.1:{port}/v1/models")
            if isinstance(data, dict) and data.get("data"):
                content.endpoint = f"http://127.0.0.1:{port}"
                content.extra["served_id"] = data["data"][0].get("id")
                content.confidence = Confidence.CONFIRMED
                break
        return content


def _ports_from_cmdline(cmd: str) -> list[int]:
    import re
    return [int(m.group(1)) for m in re.finditer(r"--port[= ]\s*(\d+)", cmd)]
