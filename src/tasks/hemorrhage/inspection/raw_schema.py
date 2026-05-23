"""Raw file schema and null statistics."""

from __future__ import annotations

from typing import List

import pandas as pd

from src.tasks.hemorrhage.io.excel_loader import ExcelLoadReport


def raw_schema_summary(
    df: pd.DataFrame,
    *,
    source_label: str,
    load_report: ExcelLoadReport,
) -> pd.DataFrame:
    """One row per column with dtype, nulls, duplicates among column names."""
    rows: List[dict] = []
    col_names = list(df.columns)
    name_counts = pd.Series(col_names).value_counts()
    duplicate_name_cols = set(name_counts[name_counts > 1].index.astype(str))

    for col in col_names:
        series = df[col]
        null_n = int(series.isna().sum())
        empty_n = 0
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            empty_n = int(series.fillna("").astype(str).str.strip().eq("").sum())
        rows.append(
            {
                "source_label": source_label,
                "file_name": load_report.path.name,
                "sheet_name": load_report.sheet_name,
                "row_count": load_report.row_count,
                "column_name": col,
                "dtype": str(series.dtype),
                "null_count": null_n,
                "null_pct": round(100.0 * null_n / len(df), 2) if len(df) else 0.0,
                "empty_string_count": empty_n,
                "nunique_non_null": int(series.nunique(dropna=True)),
                "duplicate_column_name": col in duplicate_name_cols,
                "sample_value": _safe_sample(series),
            }
        )
    return pd.DataFrame(rows)


def _safe_sample(series: pd.Series, max_len: int = 120) -> str:
    for v in series:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s:
            return s[:max_len]
    return ""
