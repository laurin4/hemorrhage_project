"""Load clinical cases from configured reports Excel."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.core.case.models import CaseConstructionStats, ClinicalCase
from src.tasks.hemorrhage.config import (
    REPORTS_XLSX_ALTERNATE_FILENAMES,
    configured_reports_xlsx_path,
    reports_sheet_name,
)
from src.tasks.hemorrhage.io.column_normalize import normalize_dataframe_columns
from src.tasks.hemorrhage.io.excel_loader import load_excel_raw
from src.tasks.hemorrhage.io.path_resolve import resolve_raw_input_path
from src.tasks.hemorrhage.preprocessing.case_builder import build_cases_from_dataframe


def load_reports_dataframe(reports_path: Path | None = None) -> Tuple[pd.DataFrame, Path, List[str]]:
    """Load and return raw reports DataFrame from configured path."""
    errors: List[str] = []
    configured = reports_path or configured_reports_xlsx_path()
    resolved = resolve_raw_input_path(configured, REPORTS_XLSX_ALTERNATE_FILENAMES, context="cases")
    if resolved.resolution == "missing":
        errors.append(f"Reports file missing: {configured}")
        return pd.DataFrame(), resolved.resolved_path, errors

    df, load_report = load_excel_raw(
        resolved.resolved_path,
        source_label="hemorrhage_reports",
        sheet_name=reports_sheet_name(),
    )
    errors.extend(load_report.errors)
    if not df.empty:
        df, _ = normalize_dataframe_columns(df, source_label="hemorrhage_reports")
    return df, resolved.resolved_path, errors


def load_clinical_cases(
    reports_path: Path | None = None,
) -> Tuple[List[ClinicalCase], CaseConstructionStats, Path, List[str]]:
    """Build ``ClinicalCase`` list from configured reports Excel."""
    df, path, errors = load_reports_dataframe(reports_path)
    if df.empty:
        return [], CaseConstructionStats(), path, errors
    cases, stats = build_cases_from_dataframe(df)
    return cases, stats, path, errors
