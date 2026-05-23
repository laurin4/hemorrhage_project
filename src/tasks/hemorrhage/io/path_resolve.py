"""
Resolve configured raw input paths; optional alternates are tried only with explicit logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedPath:
    configured_path: Path
    resolved_path: Path
    resolution: str  # found_configured | found_alternate | missing
    alternate_used: str = ""


def resolve_raw_input_path(
    configured_path: Path,
    alternate_filenames: Sequence[str],
    *,
    context: str,
) -> ResolvedPath:
    """
    Return *configured_path* if it exists; else try alternates in *same directory*.

    Does not rename files on disk.
    """
    if configured_path.exists():
        return ResolvedPath(
            configured_path=configured_path,
            resolved_path=configured_path,
            resolution="found_configured",
        )

    parent = configured_path.parent
    for name in alternate_filenames:
        if name == configured_path.name:
            continue
        candidate = parent / name
        if candidate.exists():
            LOGGER.warning(
                "[%s] Configured file not found: %s — using alternate: %s",
                context,
                configured_path,
                candidate,
            )
            return ResolvedPath(
                configured_path=configured_path,
                resolved_path=candidate,
                resolution="found_alternate",
                alternate_used=name,
            )

    LOGGER.error("[%s] Raw input missing: %s (alternates tried: %s)", context, configured_path, list(alternate_filenames))
    return ResolvedPath(
        configured_path=configured_path,
        resolved_path=configured_path,
        resolution="missing",
    )
