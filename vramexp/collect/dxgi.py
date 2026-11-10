"""Enumerazione degli adapter grafici via DXGI, per mappare LUID -> GPU.

Serve a fare il join fra i due mondi da cui prendiamo i dati:

  * i contatori PDH identificano l'adapter con un **LUID** (`luid_0x0_0x12A56`);
  * NVML identifica la GPU con un **indice** e un PCI bus id.

Nessuna delle due API espone l'identificatore dell'altra. DXGI e' il ponte: la
`DXGI_ADAPTER_DESC1` contiene sia il LUID sia descrizione, VendorId/DeviceId e
memoria dedicata, con cui si riconosce senza ambiguita' quale GPU NVML sia.

Implementato con ctypes e chiamate dirette alle vtable COM: l'alternativa
(`comtypes`) aggiungerebbe una dipendenza per tre metodi.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDOR_INTEL = 0x8086
VENDOR_MICROSOFT = 0x1414   # Microsoft Basic Render Driver / WARP

DXGI_ADAPTER_FLAG_SOFTWARE = 2


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", ctypes.c_long)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_str(cls, s: str) -> "_GUID":
        s = s.strip("{}")
        parts = s.split("-")
        d4 = bytes.fromhex(parts[3] + parts[4])
        return cls(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16),
                   (ctypes.c_ubyte * 8)(*d4))


IID_IDXGIFactory1 = _GUID.from_str("770aae78-f26f-4dba-a829-253c83d1b387")

# Indici nella vtable. IUnknown occupa 0-2, IDXGIObject 3-6, IDXGIFactory 7-11,
# IDXGIFactory1 aggiunge EnumAdapters1 in 12.
_VT_RELEASE = 2
_VT_ENUM_ADAPTERS1 = 12
# Su IDXGIAdapter1: IUnknown 0-2, IDXGIObject 3-6, IDXGIAdapter 7-9, GetDesc1 = 10.
_VT_ADAPTER_GET_DESC1 = 10

DXGI_ERROR_NOT_FOUND = 0x887A0002


@dataclass(slots=True)
class DxgiAdapter:
    luid: str                 # forma canonica, come da pdh_source.normalize_luid
    description: str
    vendor_id: int
    device_id: int
    dedicated_video_bytes: int
    shared_system_bytes: int
    is_software: bool

    @property
    def is_nvidia(self) -> bool:
        return self.vendor_id == VENDOR_NVIDIA

    @property
    def is_integrated(self) -> bool:
        # Euristica sufficiente allo scopo: le iGPU Intel/AMD hanno memoria
        # dedicata nulla o simbolica, le discrete ne hanno gigabyte.
        return self.dedicated_video_bytes < 512 * 1024 * 1024


def _call(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    """Recupera un metodo dalla vtable COM dell'oggetto."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0]
    fn_ptr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(fn_ptr)


def enum_adapters() -> list[DxgiAdapter]:
    """Elenca gli adapter grafici. Lista vuota se DXGI non e' disponibile.

    Non solleva mai: in assenza di DXGI il resto del programma deve poter
    continuare a funzionare con un solo adapter assunto.
    """
    out: list[DxgiAdapter] = []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        return out

    dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    dxgi.CreateDXGIFactory1.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory)) != 0:
        return out

    try:
        enum_fn = _call(factory, _VT_ENUM_ADAPTERS1, ctypes.c_long,
                        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_fn(factory, i, ctypes.byref(adapter)) & 0xFFFFFFFF
            if hr != 0:
                break   # DXGI_ERROR_NOT_FOUND: fine dell'enumerazione
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc = _call(adapter, _VT_ADAPTER_GET_DESC1, ctypes.c_long,
                                 ctypes.POINTER(_DXGI_ADAPTER_DESC1))
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    luid = f"0x{desc.AdapterLuid.HighPart & 0xFFFFFFFF:08X}_0x{desc.AdapterLuid.LowPart:08X}"
                    out.append(DxgiAdapter(
                        luid=luid,
                        description=desc.Description,
                        vendor_id=desc.VendorId,
                        device_id=desc.DeviceId,
                        dedicated_video_bytes=int(desc.DedicatedVideoMemory),
                        shared_system_bytes=int(desc.SharedSystemMemory),
                        is_software=bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                    ))
            finally:
                _call(adapter, _VT_RELEASE, ctypes.c_ulong)(adapter)
            i += 1
    finally:
        _call(factory, _VT_RELEASE, ctypes.c_ulong)(factory)

    return out


def map_luid_to_nvml(adapters: list[DxgiAdapter],
                     nvml_gpus: list[tuple[int, str, int]]) -> dict[str, int]:
    """Associa ogni LUID DXGI all'indice NVML della stessa GPU.

    `nvml_gpus` e' una lista di (index, name, total_bytes).

    Il match si basa su nome e memoria totale. DXGI riporta la memoria dedicata
    leggermente inferiore a NVML (parte e' riservata al firmware), quindi il
    confronto usa una tolleranza del 10% invece dell'uguaglianza.
    """
    result: dict[str, int] = {}
    used: set[int] = set()
    nv_adapters = [a for a in adapters if a.is_nvidia and not a.is_software]

    for a in nv_adapters:
        best_idx, best_score = None, -1.0
        for idx, name, total in nvml_gpus:
            if idx in used:
                continue
            score = 0.0
            if name and a.description:
                # "NVIDIA GeForce RTX 5090" da entrambe le parti, ma DXGI a volte
                # tronca o varia il prefisso: confronto per token in comune.
                tok_a = set(a.description.lower().split())
                tok_b = set(name.lower().split())
                if tok_a & tok_b:
                    score += len(tok_a & tok_b) / max(1, len(tok_a | tok_b))
            if total > 0 and a.dedicated_video_bytes > 0:
                ratio = a.dedicated_video_bytes / total
                if 0.90 <= ratio <= 1.10:
                    score += 1.0
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is not None and best_score > 0:
            result[a.luid] = best_idx
            used.add(best_idx)

    # Caso degenere ma frequentissimo: una sola GPU NVIDIA e un solo adapter
    # NVIDIA. Se l'euristica non ha trovato il match, forzarlo e' corretto.
    if not result and len(nv_adapters) == 1 and len(nvml_gpus) == 1:
        result[nv_adapters[0].luid] = nvml_gpus[0][0]

    return result
