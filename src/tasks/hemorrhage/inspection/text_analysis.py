"""Text field missingness and length statistics (no NLP)."""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from src.core.case.models import ClinicalCase
from src.tasks.hemorrhage.constants import HEMORRHAGE_TEXT_FIELDS


def _text_len(value: object) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return len(str(value).strip())


def text_field_statistics(reports_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    n = len(reports_df)
    for field in HEMORRHAGE_TEXT_FIELDS:
        if field not in reports_df.columns:
            rows.append(
                {
                    "field": field,
                    "present": False,
                    "row_count": n,
                    "null_or_missing_column": True,
                    "empty_count": n if n else 0,
                    "mean_length": 0,
                    "median_length": 0,
                    "max_length": 0,
                    "min_length_non_empty": 0,
                    "duplicate_text_count": 0,
                }
            )
            continue

        series = reports_df[field]
        lengths = series.map(_text_len)
        empty = int((lengths == 0).sum())
        non_empty = lengths[lengths > 0]
        dup_text = 0
        non_empty_texts = series.map(lambda v: str(v).strip() if _text_len(v) else None).dropna()
        if len(non_empty_texts):
            dup_text = int(non_empty_texts.duplicated().sum())

        rows.append(
            {
                "field": field,
                "present": True,
                "row_count": n,
                "null_or_missing_column": False,
                "empty_count": empty,
                "empty_pct": round(100.0 * empty / n, 2) if n else 0,
                "mean_length": round(float(non_empty.mean()), 1) if len(non_empty) else 0,
                "median_length": float(non_empty.median()) if len(non_empty) else 0,
                "max_length": int(non_empty.max()) if len(non_empty) else 0,
                "min_length_non_empty": int(non_empty.min()) if len(non_empty) else 0,
                "duplicate_text_count": dup_text,
            }
        )
    return pd.DataFrame(rows)


def text_field_samples(
    reports_df: pd.DataFrame,
    *,
    n_per_field: int = 3,
) -> pd.DataFrame:
    """Short samples for manual review (not full export)."""
    rows: List[dict] = []
    for field in HEMORRHAGE_TEXT_FIELDS:
        if field not in reports_df.columns:
            continue
        non_empty = reports_df[reports_df[field].map(_text_len) > 0].head(n_per_field)
        for idx, row in non_empty.iterrows():
            text = str(row[field]).strip()
            rows.append(
                {
                    "field": field,
                    "row_index": int(idx),
                    "excel_pid": row.get("excel_pid", ""),
                    "typus": row.get("typus", ""),
                    "text_preview": text[:500],
                    "text_length": len(text),
                }
            )
    return pd.DataFrame(rows)


def case_text_length_table(cases: Sequence[ClinicalCase]) -> pd.DataFrame:
    rows = []
    for c in cases:
        st = c.structured_case_text()
        rows.append(
            {
                "case_id": c.case_id,
                "structured_case_text_length": len(st),
                "n_reports_available": c.n_reports_available,
                "is_complete": c.is_complete,
            }
        )
    return pd.DataFrame(rows)
