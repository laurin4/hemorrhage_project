"""
Robust Excel loading for hemorrhage raw inputs.

Preserves row counts; logs sheet selection and dtypes; does not drop rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class ExcelLoadReport:
    path: Path
    source_label: str
    sheet_name: str
    sheet_names: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"{self.source_label}: path={self.path.name} sheet={self.sheet_name!r} "
            f"rows={self.row_count} cols={self.column_count}"
        )


def load_excel_raw(
    path: Path,
    *,
    source_label: str,
    sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, ExcelLoadReport]:
    """
    Load one Excel workbook sheet into a DataFrame.

    - Uses ``openpyxl`` engine.
    - All columns read as object first pass would lose types — use default inference
      but case keys are normalized later as strings.
  - Empty file returns empty DataFrame with logged warning.
    """
    report = ExcelLoadReport(path=path, source_label=source_label, sheet_name="")

    if not path.exists():
        report.errors.append(f"file_not_found: {path}")
        return pd.DataFrame(), report

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        report.errors.append(f"excel_open_failed: {exc}")
        LOGGER.exception("Failed to open Excel %s", path)
        return pd.DataFrame(), report

    report.sheet_names = list(xl.sheet_names)
    use_sheet = sheet_name
    if use_sheet is None:
        use_sheet = xl.sheet_names[0] if xl.sheet_names else ""
    elif use_sheet not in xl.sheet_names:
        report.errors.append(f"sheet_not_found: {use_sheet!r} in {report.sheet_names}")
        LOGGER.error(
            "[%s] Sheet %r not in workbook %s (%s)",
            source_label,
            use_sheet,
            path.name,
            report.sheet_names,
        )
        return pd.DataFrame(), report

    report.sheet_name = use_sheet

    try:
        df = pd.read_excel(xl, sheet_name=use_sheet, dtype=object)
    except Exception as exc:
        report.errors.append(f"read_failed: {exc}")
        LOGGER.exception("Failed to read sheet %s from %s", use_sheet, path)
        return pd.DataFrame(), report

    # Preserve row index alignment with Excel (0..n-1)
    df = df.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    report.row_count = len(df)
    report.column_count = len(df.columns)
    report.columns = list(df.columns)

    LOGGER.info("[%s] %s", source_label, report.summary_line())
    if df.empty:
        LOGGER.warning("[%s] Loaded 0 rows from %s", source_label, path)

    return df, report
