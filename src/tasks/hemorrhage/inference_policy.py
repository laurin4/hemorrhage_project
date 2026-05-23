"""
Hemorrhage inference policy (Phase 0).

Prefilter / auto-negative behavior from delirium MUST NOT apply to hemorrhage
until task-specific evidence rules exist.
"""

from __future__ import annotations

import os

# disabled: never skip inference due to missing keywords (default for hemorrhage prep)
# relaxed: reserved for future partial prefilter
# delirium_legacy: use delirium evidence prefilter (NOT for production hemorrhage)
PREFILTER_MODE_ENV = "HEMORRHAGE_PREFILTER_MODE"
DEFAULT_PREFILTER_MODE = "disabled"


def hemorrhage_prefilter_mode() -> str:
    raw = os.environ.get(PREFILTER_MODE_ENV, DEFAULT_PREFILTER_MODE).strip().lower()
    if raw in ("disabled", "relax", "relaxed", "delirium_legacy"):
        return "relaxed" if raw in ("relax", "relaxed") else raw
    return DEFAULT_PREFILTER_MODE


def prefilter_skip_allowed() -> bool:
    """When False, pipeline must not assign auto-negative solely due to missing keyword hits."""
    return hemorrhage_prefilter_mode() == "delirium_legacy"


def prefilter_status_label() -> str:
    mode = hemorrhage_prefilter_mode()
    if mode == "disabled":
        return "hemorrhage_prefilter_disabled"
    if mode == "relaxed":
        return "hemorrhage_prefilter_relaxed"
    return "delirium_legacy_prefilter"
