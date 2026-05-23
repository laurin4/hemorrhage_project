"""Exploratory keyword counts — no classification."""

from __future__ import annotations

from typing import List

import pandas as pd

from src.tasks.hemorrhage.constants import HEMORRHAGE_TEXT_FIELDS, KEYWORD_EXPLORATION_TERMS


def keyword_exploration_table(reports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count rows containing each term (case-insensitive) per text field.

    Also reports fraction of non-empty rows — for leakage / explicitness estimates.
    """
    rows: List[dict] = []
    for field in HEMORRHAGE_TEXT_FIELDS:
        if field not in reports_df.columns:
            for term in KEYWORD_EXPLORATION_TERMS:
                rows.append(
                    {
                        "field": field,
                        "keyword": term,
                        "column_present": False,
                        "rows_with_term": 0,
                        "rows_non_empty": 0,
                        "pct_of_non_empty": 0.0,
                        "pct_of_all_rows": 0.0,
                    }
                )
            continue

        series = reports_df[field].fillna("").astype(str)
        non_empty_mask = series.str.strip().astype(bool)
        n_all = len(series)
        n_ne = int(non_empty_mask.sum())

        for term in KEYWORD_EXPLORATION_TERMS:
            pattern = term.lower()
            hits = series.str.lower().str.contains(pattern, regex=False, na=False)
            n_hits = int(hits.sum())
            rows.append(
                {
                    "field": field,
                    "keyword": term,
                    "column_present": True,
                    "rows_with_term": n_hits,
                    "rows_non_empty": n_ne,
                    "pct_of_non_empty": round(100.0 * n_hits / n_ne, 2) if n_ne else 0.0,
                    "pct_of_all_rows": round(100.0 * n_hits / n_all, 2) if n_all else 0.0,
                }
            )
    return pd.DataFrame(rows)
