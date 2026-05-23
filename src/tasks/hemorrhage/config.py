"""
Hemorrhage task path configuration (filenames + env overrides).

Physical files on disk are never renamed — only explicit configured paths or
documented alternate filenames (e.g. space vs underscore) are resolved with logging.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.pipeline.paths import PROJECT_ROOT, REAL_RAW_DIR

# Default on-disk names (underscore form; spaces allowed via alternates)
# REPORTS = clinical export (pidlist / OP-Eintritt-Austritt text)
DEFAULT_REPORTS_XLSX_FILENAME = "NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx"
# REFERENCE = manual labels / verification (CCM DAVF cohort sheet)
DEFAULT_REFERENCE_XLSX_FILENAME = "260507_CCM_DAVF.xlsx"

# Alternate filenames if the configured path is missing (not a silent rename)
REPORTS_XLSX_ALTERNATE_FILENAMES: tuple[str, ...] = (
    DEFAULT_REPORTS_XLSX_FILENAME,
)

REFERENCE_XLSX_ALTERNATE_FILENAMES: tuple[str, ...] = (
    "260507 CCM DAVF.xlsx",
    DEFAULT_REFERENCE_XLSX_FILENAME,
)


def _path_from_env_or_default(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    p = Path(raw)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def configured_reports_xlsx_path() -> Path:
    """Primary configured path for clinical report export (NCH pidlist / Berichte export)."""
    return _path_from_env_or_default(
        "HEMORRHAGE_REPORTS_XLSX",
        REAL_RAW_DIR / DEFAULT_REPORTS_XLSX_FILENAME,
    )


def configured_reference_xlsx_path() -> Path:
    """Primary configured path for reference / manual labels (260507 CCM DAVF)."""
    return _path_from_env_or_default(
        "HEMORRHAGE_REFERENCE_XLSX",
        REAL_RAW_DIR / DEFAULT_REFERENCE_XLSX_FILENAME,
    )


def reports_sheet_name() -> str | None:
    """Optional sheet name for reports workbook; first sheet if unset."""
    raw = os.environ.get("HEMORRHAGE_REPORTS_SHEET", "").strip()
    return raw or None


def reference_sheet_name() -> str | None:
    raw = os.environ.get("HEMORRHAGE_REFERENCE_SHEET", "").strip()
    return raw or None
