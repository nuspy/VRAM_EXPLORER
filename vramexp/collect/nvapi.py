"""Accesso a NVAPI via ctypes, per la Driver Settings Repository (DRS).

NVAPI non esporta le sue funzioni per nome: espone un solo simbolo,
`nvapi_QueryInterface`, che dato un ID numerico restituisce il puntatore alla
funzione. Gli ID sono costanti stabili fra le versioni del driver.

Serve per una cosa in particolare: la **CUDA Sysmem Fallback Policy**, cioe' la
scelta fra "quando la VRAM finisce, continua usando la RAM di sistema" (default)
e "quando la VRAM finisce, fai fallire l'allocazione". La seconda e' spesso
preferibile per il lavoro serio: un training che fallisce in due secondi si
riavvia con parametri corretti, mentre uno che degrada di 50 volte puo' bruciare
un'intera notte prima che qualcuno se ne accorga.

Il valore aggiunto rispetto al pannello NVIDIA e' il supporto ai **profili per
applicazione**: si puo' imporre il fail-fast solo a `python.exe` lasciando il
comportamento permissivo a giochi e browser.

Nota sui privilegi: leggere le impostazioni funziona da utente normale; salvarle
scrive in HKLM e richiede elevazione (vedi `control/fallback_policy.py`).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

# --- ID di interfaccia, stabili fra le versioni del driver -----------------
FN_IDS = {
    "NvAPI_Initialize": 0x0150E828,
    "NvAPI_Unload": 0xD22BDD7E,
    "NvAPI_GetErrorMessage": 0x6C2D048C,
    "NvAPI_DRS_CreateSession": 0x0694D52E,
    "NvAPI_DRS_DestroySession": 0xDAD9CFF8,
    "NvAPI_DRS_LoadSettings": 0x375DBD6B,
    "NvAPI_DRS_SaveSettings": 0xFCBC7E14,
    "NvAPI_DRS_GetBaseProfile": 0xDA8466A0,
    "NvAPI_DRS_GetSetting": 0x73BF8338,
    "NvAPI_DRS_SetSetting": 0x577DD202,
    "NvAPI_DRS_FindProfileByName": 0x7E4A9A0B,
    "NvAPI_DRS_CreateProfile": 0xCC176068,
    "NvAPI_DRS_CreateApplication": 0x4347A9DE,
    "NvAPI_DRS_FindApplicationByName": 0xEEE566B2,
    "NvAPI_DRS_DeleteProfile": 0x17093206,
    "NvAPI_GPU_GetMemoryInfoEx": 0xC0599498,
}

# Impostazione che ci interessa. 0 = comportamento predefinito del driver
# (consente lo spostamento su RAM), 1 = nessun fallback (l'allocazione fallisce).
CUDA_SYSMEM_FALLBACK_POLICY_ID = 0x10FC2391
SYSMEM_FALLBACK_DRIVER_DEFAULT = 0x00000000
SYSMEM_FALLBACK_PREFER_NO_SYSMEM = 0x00000001

NVAPI_OK = 0
NVAPI_INCOMPATIBLE_STRUCT_VERSION = -9
# Verificato sul driver 610.62: scrivere le impostazioni DRS da un processo non
# elevato restituisce -137. (-127 e' SYSWOW64_NOT_SUPPORTED, da non confondere.)
NVAPI_INVALID_USER_PRIVILEGE = -137
# Codici della famiglia DRS. Attenzione all'ordine: -160 e' SETTING_NOT_FOUND,
# che e' l'esito **normale** quando l'utente non ha mai toccato l'impostazione,
# non un errore da segnalare.
NVAPI_SETTING_NOT_FOUND = -160
NVAPI_SETTING_SIZE_TOO_LARGE = -161
NVAPI_SERVICE_NOT_CONFIGURED = -162
NVAPI_PROFILE_NOT_FOUND = -163
NVAPI_PROFILE_NAME_IN_USE = -164
NVAPI_PROFILE_NAME_EMPTY = -165
NVAPI_EXECUTABLE_NOT_FOUND = -166
NVAPI_EXECUTABLE_ALREADY_IN_USE = -167

NVAPI_UNICODE_STRING_MAX = 2048
NVAPI_BINARY_DATA_MAX = 4096

NVDRS_SETTING_TYPE_DWORD = 0
NVDRS_SETTING_LOCATION_CURRENT_PROFILE = 0


class _NvAPI_UnicodeString(ctypes.Structure):
    _fields_ = [("data", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX)]


class _NVDRS_BINARY(ctypes.Structure):
    _fields_ = [("valueLength", ctypes.c_uint),
                ("valueData", ctypes.c_ubyte * NVAPI_BINARY_DATA_MAX)]


class _NVDRS_SETTING_VALUE_UNION(ctypes.Union):
    _fields_ = [("u32Value", ctypes.c_uint),
                ("binaryValue", _NVDRS_BINARY),
                ("wszValue", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX)]


class _NVDRS_SETTING(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("settingName", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
        ("settingId", ctypes.c_uint),
        ("settingType", ctypes.c_uint),
        ("settingLocation", ctypes.c_uint),
        ("isCurrentPredefined", ctypes.c_uint),
        ("isPredefinedValid", ctypes.c_uint),
        ("predefinedValue", _NVDRS_SETTING_VALUE_UNION),
        ("currentValue", _NVDRS_SETTING_VALUE_UNION),
    ]


class _NVDRS_PROFILE(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("profileName", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
        ("gpuSupport", ctypes.c_uint * 32),
        ("isPredefined", ctypes.c_uint),
        ("numOfApps", ctypes.c_uint),
        ("numOfSettings", ctypes.c_uint),
    ]


class _NVDRS_APPLICATION(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("isPredefined", ctypes.c_uint),
        ("appName", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
        ("userFriendlyName", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
        ("launcher", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
        ("fileInFolder", wintypes.WCHAR * NVAPI_UNICODE_STRING_MAX),
    ]


def _make_version(struct_type: type[ctypes.Structure], version: int) -> int:
    """NVAPI codifica versione e dimensione della struttura in un solo campo.

    Se il valore non corrisponde esattamente a quanto il driver si aspetta, la
    chiamata fallisce con INCOMPATIBLE_STRUCT_VERSION: e' il modo con cui NVAPI
    mantiene la compatibilita' binaria fra versioni dell'SDK.
    """
    return ctypes.sizeof(struct_type) | (version << 16)


@dataclass
class SettingValue:
    setting_id: int
    value: int
    is_predefined: bool
    predefined_value: int | None
    profile: str


class NvApiError(RuntimeError):
    def __init__(self, func: str, status: int, hint: str = "",
                 driver_message: str | None = None) -> None:
        self.status = status
        msg = f"{func} fallita (status {status}"
        if driver_message:
            msg += f", {driver_message}"
        msg += ")"
        if status == NVAPI_INVALID_USER_PRIVILEGE:
            msg += " — privilegi insufficienti: serve l'elevazione"
        if hint:
            msg += f": {hint}"
        super().__init__(msg)


class NvApi:
    """Wrapper minimale su NVAPI, limitato a cio' che serve al progetto."""

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self._fn: dict[str, ctypes.CFUNCTYPE] = {}
        self._dll = None
        try:
            self._dll = ctypes.WinDLL("nvapi64.dll")
        except OSError as e:
            self.error = f"nvapi64.dll non caricabile: {e}"
            return

        q = self._dll.nvapi_QueryInterface
        q.restype = ctypes.c_void_p
        q.argtypes = [ctypes.c_uint]
        self._query = q

        init = self._proto("NvAPI_Initialize", ctypes.c_int)
        if init is None:
            self.error = "NvAPI_Initialize non risolvibile"
            return
        status = init()
        if status != NVAPI_OK:
            self.error = f"NvAPI_Initialize ha restituito {status}"
            return
        self.available = True

    def error_message(self, status: int) -> str | None:
        """Descrizione testuale di uno stato, chiesta al driver stesso.

        Preferibile a una tabella di costanti compilata a mano: i codici NVAPI
        sono numerosi e facili da trascrivere male, e il driver conosce quelli
        della propria versione.
        """
        fn = self._proto("NvAPI_GetErrorMessage", ctypes.c_int,
                         ctypes.c_int, ctypes.c_char * 64)
        if fn is None:
            return None
        buf = (ctypes.c_char * 64)()
        if fn(status, buf) != NVAPI_OK:
            return None
        return buf.value.decode("ascii", errors="replace") or None

    def _raise(self, func: str, status: int, hint: str = "") -> None:
        raise NvApiError(func, status, hint, self.error_message(status))

    def _proto(self, name: str, restype, *argtypes):
        if name in self._fn:
            return self._fn[name]
        fid = FN_IDS.get(name)
        if fid is None:
            return None
        ptr = self._query(fid)
        if not ptr:
            return None
        fn = ctypes.WINFUNCTYPE(restype, *argtypes)(ptr)
        self._fn[name] = fn
        return fn

    # -- sessione DRS ------------------------------------------------------
    def _session(self) -> ctypes.c_void_p:
        create = self._proto("NvAPI_DRS_CreateSession", ctypes.c_int,
                             ctypes.POINTER(ctypes.c_void_p))
        load = self._proto("NvAPI_DRS_LoadSettings", ctypes.c_int, ctypes.c_void_p)
        if create is None or load is None:
            raise NvApiError("DRS_CreateSession", -1, "interfaccia non disponibile")
        h = ctypes.c_void_p()
        st = create(ctypes.byref(h))
        if st != NVAPI_OK:
            self._raise("NvAPI_DRS_CreateSession", st)
        st = load(h)
        if st != NVAPI_OK:
            self._destroy(h)
            self._raise("NvAPI_DRS_LoadSettings", st)
        return h

    def _destroy(self, h: ctypes.c_void_p) -> None:
        fn = self._proto("NvAPI_DRS_DestroySession", ctypes.c_int, ctypes.c_void_p)
        if fn is not None:
            fn(h)

    def _base_profile(self, h: ctypes.c_void_p) -> ctypes.c_void_p:
        fn = self._proto("NvAPI_DRS_GetBaseProfile", ctypes.c_int, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p))
        p = ctypes.c_void_p()
        st = fn(h, ctypes.byref(p))
        if st != NVAPI_OK:
            self._raise("NvAPI_DRS_GetBaseProfile", st)
        return p

    # -- lettura/scrittura di impostazioni ---------------------------------
    def get_setting(self, setting_id: int = CUDA_SYSMEM_FALLBACK_POLICY_ID,
                    profile_name: str | None = None) -> SettingValue | None:
        """Legge un'impostazione dal profilo globale o da uno per applicazione.

        `None` significa "non impostata esplicitamente", che e' diverso da
        "impostata al valore predefinito": la distinzione conta nella GUI, dove
        vogliamo mostrare se l'utente ha gia' toccato la manopola.
        """
        if not self.available:
            return None
        h = self._session()
        try:
            profile = (self._find_profile(h, profile_name) if profile_name
                       else self._base_profile(h))
            if profile is None:
                return None
            fn = self._proto("NvAPI_DRS_GetSetting", ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_uint,
                             ctypes.POINTER(_NVDRS_SETTING))
            setting = _NVDRS_SETTING()
            setting.version = _make_version(_NVDRS_SETTING, 1)
            st = fn(h, profile, setting_id, ctypes.byref(setting))
            if st == NVAPI_SETTING_NOT_FOUND:
                return None
            if st != NVAPI_OK:
                self._raise("NvAPI_DRS_GetSetting", st)
            return SettingValue(
                setting_id=setting_id,
                value=int(setting.currentValue.u32Value),
                is_predefined=bool(setting.isCurrentPredefined),
                predefined_value=(int(setting.predefinedValue.u32Value)
                                  if setting.isPredefinedValid else None),
                profile=profile_name or "<globale>",
            )
        finally:
            self._destroy(h)

    def set_setting(self, value: int, setting_id: int = CUDA_SYSMEM_FALLBACK_POLICY_ID,
                    profile_name: str | None = None,
                    application_exe: str | None = None) -> None:
        """Imposta e salva. Solleva `NvApiError` se il salvataggio e' negato.

        Con `application_exe` l'impostazione viene applicata a un profilo
        dedicato a quell'eseguibile, creandolo se non esiste, invece che
        globalmente a tutto il sistema.
        """
        if not self.available:
            raise NvApiError("NvAPI", -1, self.error or "NVAPI non disponibile")
        h = self._session()
        try:
            if application_exe:
                profile = self._ensure_app_profile(h, profile_name or _profile_for(application_exe),
                                                   application_exe)
            elif profile_name:
                profile = self._find_profile(h, profile_name)
                if profile is None:
                    self._raise("NvAPI_DRS_FindProfileByName", NVAPI_PROFILE_NOT_FOUND,
                                f"profilo {profile_name!r} inesistente")
            else:
                profile = self._base_profile(h)

            fn = self._proto("NvAPI_DRS_SetSetting", ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.POINTER(_NVDRS_SETTING))
            setting = _NVDRS_SETTING()
            setting.version = _make_version(_NVDRS_SETTING, 1)
            setting.settingId = setting_id
            setting.settingType = NVDRS_SETTING_TYPE_DWORD
            setting.settingLocation = NVDRS_SETTING_LOCATION_CURRENT_PROFILE
            setting.currentValue.u32Value = value
            st = fn(h, profile, ctypes.byref(setting))
            if st != NVAPI_OK:
                self._raise("NvAPI_DRS_SetSetting", st)

            save = self._proto("NvAPI_DRS_SaveSettings", ctypes.c_int, ctypes.c_void_p)
            st = save(h)
            if st != NVAPI_OK:
                # Il caso tipico: scrittura in HKLM senza elevazione.
                self._raise("NvAPI_DRS_SaveSettings", st,
                            "il salvataggio richiede privilegi di amministratore")
        finally:
            self._destroy(h)

    def _find_profile(self, h: ctypes.c_void_p, name: str) -> ctypes.c_void_p | None:
        fn = self._proto("NvAPI_DRS_FindProfileByName", ctypes.c_int, ctypes.c_void_p,
                         _NvAPI_UnicodeString, ctypes.POINTER(ctypes.c_void_p))
        us = _NvAPI_UnicodeString()
        us.data = name
        p = ctypes.c_void_p()
        st = fn(h, us, ctypes.byref(p))
        return p if st == NVAPI_OK else None

    def _ensure_app_profile(self, h: ctypes.c_void_p, profile_name: str,
                            exe: str) -> ctypes.c_void_p:
        profile = self._find_profile(h, profile_name)
        if profile is None:
            create = self._proto("NvAPI_DRS_CreateProfile", ctypes.c_int, ctypes.c_void_p,
                                 ctypes.POINTER(_NVDRS_PROFILE), ctypes.POINTER(ctypes.c_void_p))
            prof = _NVDRS_PROFILE()
            prof.version = _make_version(_NVDRS_PROFILE, 1)
            prof.profileName = profile_name
            p = ctypes.c_void_p()
            st = create(h, ctypes.byref(prof), ctypes.byref(p))
            if st != NVAPI_OK:
                self._raise("NvAPI_DRS_CreateProfile", st)
            profile = p

        add = self._proto("NvAPI_DRS_CreateApplication", ctypes.c_int, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.POINTER(_NVDRS_APPLICATION))
        app = _NVDRS_APPLICATION()
        app.version = _make_version(_NVDRS_APPLICATION, 1)
        app.appName = exe
        app.userFriendlyName = exe
        st = add(h, profile, ctypes.byref(app))
        # Se l'eseguibile e' gia' associato a questo profilo, l'errore e' atteso
        # e va ignorato: lo scopo e' che l'associazione esista, non crearla ora.
        if st not in (NVAPI_OK, NVAPI_EXECUTABLE_ALREADY_IN_USE):
            self._raise("NvAPI_DRS_CreateApplication", st)
        return profile

    def unload(self) -> None:
        if self.available:
            fn = self._proto("NvAPI_Unload", ctypes.c_int)
            if fn is not None:
                fn()
            self.available = False


def _profile_for(exe: str) -> str:
    import os
    return f"VramExplorer - {os.path.basename(exe)}"
