"""Policy di fallback sulla memoria di sistema: usare la RAM o fallire.

Quando la VRAM finisce, il driver NVIDIA per default continua allocando nella RAM
di sistema. L'applicazione non se ne accorge: nessun errore, nessun avviso, solo
un crollo prestazionale che puo' arrivare a due ordini di grandezza perche' ogni
accesso passa dal bus PCIe.

L'alternativa e' `Prefer No Sysmem Fallback`: l'allocazione fallisce subito con
un errore di memoria esaurita. Sembra peggio, ma quasi sempre non lo e': un
training che muore in due secondi si riavvia con batch piu' piccolo, mentre uno
che degrada di cinquanta volte puo' bruciare una notte prima che qualcuno guardi.

**Il valore aggiunto rispetto al pannello NVIDIA** e' il profilo per
applicazione: imporre il fail-fast solo a `python.exe` e ai server di
inferenza, lasciando permissivi giochi e browser, dove il degrado e' preferibile
a un crash.

**Privilegi.** La lettura funziona da utente normale. La scrittura chiama
`NvAPI_DRS_SaveSettings`, che scrive in HKLM e restituisce
`NVAPI_INVALID_USER_PRIVILEGE` (-137) senza elevazione — verificato sul driver
610.62. Il servizio gira volutamente non elevato, quindi la scrittura viene
delegata a un helper che chiede l'elevazione con un prompt UAC visibile: mai di
nascosto.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from ..collect.nvapi import (
    CUDA_SYSMEM_FALLBACK_POLICY_ID,
    SYSMEM_FALLBACK_DRIVER_DEFAULT,
    SYSMEM_FALLBACK_PREFER_NO_SYSMEM,
    NvApi,
    NvApiError,
)

log = logging.getLogger("vramexp.policy")

POLICY_LABELS = {
    SYSMEM_FALLBACK_DRIVER_DEFAULT: "Consenti l'uso della RAM di sistema (predefinito del driver)",
    SYSMEM_FALLBACK_PREFER_NO_SYSMEM: "Fai fallire l'allocazione invece di usare la RAM",
}

POLICY_EXPLAIN = {
    SYSMEM_FALLBACK_DRIVER_DEFAULT: (
        "Quando la VRAM si esaurisce il driver continua allocando nella RAM di sistema. "
        "L'applicazione non riceve errori ma puo' rallentare di 10-100 volte."),
    SYSMEM_FALLBACK_PREFER_NO_SYSMEM: (
        "Quando la VRAM si esaurisce l'allocazione fallisce subito con un errore di "
        "memoria insufficiente. Preferibile per training e inferenza, dove un errore "
        "immediato e' piu' utile di un degrado silenzioso."),
}


@dataclass
class PolicyState:
    value: int | None                 # None = mai impostata esplicitamente
    scope: str                        # "globale" oppure il nome dell'eseguibile
    effective_value: int              # cio' che il driver applica davvero
    label: str
    explanation: str
    is_default: bool
    writable: bool
    write_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "value": self.value, "scope": self.scope,
            "effective_value": self.effective_value, "label": self.label,
            "explanation": self.explanation, "is_default": self.is_default,
            "writable": self.writable, "write_hint": self.write_hint,
        }


class FallbackPolicyManager:
    def __init__(self) -> None:
        self._nvapi = NvApi()

    @property
    def available(self) -> bool:
        return self._nvapi.available

    @property
    def error(self) -> str | None:
        return self._nvapi.error

    # -- lettura -----------------------------------------------------------
    def get(self, application_exe: str | None = None) -> PolicyState:
        scope = os.path.basename(application_exe) if application_exe else "globale"
        if not self.available:
            return PolicyState(None, scope, SYSMEM_FALLBACK_DRIVER_DEFAULT,
                               "non determinabile",
                               self._nvapi.error or "NVAPI non disponibile",
                               True, False, self._nvapi.error or "")
        profile = _profile_name(application_exe) if application_exe else None
        try:
            sv = self._nvapi.get_setting(CUDA_SYSMEM_FALLBACK_POLICY_ID, profile)
        except NvApiError as e:
            log.debug("lettura policy fallita: %s", e)
            sv = None

        value = sv.value if sv else None
        # Assenza dell'impostazione significa che vale il default del driver,
        # che e' quello permissivo.
        effective = value if value is not None else SYSMEM_FALLBACK_DRIVER_DEFAULT
        return PolicyState(
            value=value, scope=scope, effective_value=effective,
            label=POLICY_LABELS.get(effective, f"valore sconosciuto ({effective})"),
            explanation=POLICY_EXPLAIN.get(effective, ""),
            is_default=(value is None),
            writable=True,
            write_hint=("la modifica richiede l'elevazione: verra' mostrato "
                        "un prompt di Controllo account utente"),
        )

    def list_scopes(self, exes: list[str]) -> list[dict]:
        """Stato globale piu' quello di ogni eseguibile indicato."""
        out = [self.get(None).to_dict()]
        for exe in exes:
            out.append(self.get(exe).to_dict())
        return out

    # -- scrittura ---------------------------------------------------------
    def set(self, value: int, application_exe: str | None = None,
            allow_elevation: bool = True) -> tuple[bool, str]:
        """Imposta la policy. Prova in-process, poi con elevazione se serve."""
        if value not in (SYSMEM_FALLBACK_DRIVER_DEFAULT, SYSMEM_FALLBACK_PREFER_NO_SYSMEM):
            return False, f"valore non valido: {value}"
        if not self.available:
            return False, self._nvapi.error or "NVAPI non disponibile"

        try:
            self._nvapi.set_setting(value, CUDA_SYSMEM_FALLBACK_POLICY_ID,
                                    application_exe=application_exe)
            return True, self._describe(value, application_exe)
        except NvApiError as e:
            if not _needs_elevation(e) or not allow_elevation:
                return False, str(e)
            log.info("policy: scrittura diretta negata, si tenta con elevazione")

        ok, msg = _set_elevated(value, application_exe)
        if ok:
            return True, self._describe(value, application_exe) + " (tramite elevazione)"
        return False, msg

    def _describe(self, value: int, exe: str | None) -> str:
        scope = f"per {os.path.basename(exe)}" if exe else "globalmente"
        return f"policy impostata {scope}: {POLICY_LABELS[value]}"

    def close(self) -> None:
        self._nvapi.unload()


def _profile_name(exe: str) -> str:
    return f"VramExplorer - {os.path.basename(exe)}"


def _needs_elevation(e: NvApiError) -> bool:
    from ..collect.nvapi import NVAPI_INVALID_USER_PRIVILEGE
    return e.status == NVAPI_INVALID_USER_PRIVILEGE


def _set_elevated(value: int, application_exe: str | None) -> tuple[bool, str]:
    """Riesegue la scrittura in un processo elevato.

    Si usa `ShellExecuteW` con il verbo "runas", che mostra il prompt UAC. Non
    esiste modo di elevare silenziosamente, ed e' giusto cosi': modificare le
    impostazioni del driver per tutto il sistema deve essere una scelta visibile.

    Il risultato passa da un file temporaneo perche' un processo elevato non
    condivide stdout con quello che lo ha lanciato.
    """
    result_path = os.path.join(tempfile.gettempdir(),
                               f"vramexp_policy_{os.getpid()}.json")
    if os.path.exists(result_path):
        os.unlink(result_path)

    args = [
        "-m", "vramexp.control.fallback_policy",
        "--value", str(value),
        "--result", result_path,
    ]
    if application_exe:
        args += ["--exe", application_exe]

    try:
        import win32con
        import win32event
        import win32process
        from win32comext.shell import shellcon
        from win32comext.shell.shell import ShellExecuteEx

        proc = ShellExecuteEx(
            nShow=win32con.SW_HIDE,
            fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,
            lpVerb="runas",
            lpFile=sys.executable,
            lpParameters=subprocess.list2cmdline(args),
            lpDirectory=str(_repo_root()),
        )
        handle = proc["hProcess"]
        win32event.WaitForSingleObject(handle, 60000)
        code = win32process.GetExitCodeProcess(handle)
    except Exception as e:
        if "canceled" in str(e).lower() or "annull" in str(e).lower():
            return False, "elevazione annullata dall'utente"
        return False, f"elevazione fallita: {e}"

    if os.path.exists(result_path):
        try:
            with open(result_path, encoding="utf-8") as f:
                data = json.load(f)
            os.unlink(result_path)
            return bool(data.get("ok")), str(data.get("message", ""))
        except (OSError, json.JSONDecodeError):
            pass
    return (code == 0), ("impostazione applicata" if code == 0
                         else f"il processo elevato e' uscito con codice {code}")


def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def _main() -> int:
    """Punto di ingresso del processo elevato."""
    import argparse

    parser = argparse.ArgumentParser(description="Imposta la sysmem fallback policy")
    parser.add_argument("--value", type=int, required=True)
    parser.add_argument("--exe", default=None)
    parser.add_argument("--result", default=None)
    ns = parser.parse_args()

    mgr = FallbackPolicyManager()
    try:
        ok, msg = mgr.set(ns.value, ns.exe, allow_elevation=False)
    finally:
        mgr.close()

    if ns.result:
        try:
            with open(ns.result, "w", encoding="utf-8") as f:
                json.dump({"ok": ok, "message": msg}, f, ensure_ascii=False)
        except OSError:
            pass
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
