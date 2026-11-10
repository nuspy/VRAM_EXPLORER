"""Pianificazione: stima del fabbisogno, libreria modelli, profili operativi."""

from .kv_cache import VramEstimate, estimate_gguf, kv_cache_bytes, max_context_for_budget
from .library_scan import LibraryIndex, LibraryScanner, ModelEntry
from .profiles import Profile, ProfileManager, ProfileStep
from .whatif import PlanResult, Planner

__all__ = [
    "VramEstimate", "estimate_gguf", "kv_cache_bytes", "max_context_for_budget",
    "LibraryIndex", "LibraryScanner", "ModelEntry",
    "Profile", "ProfileManager", "ProfileStep",
    "PlanResult", "Planner",
]
