"""
Case-level export schema (one row = one case).

Phase 0: placeholders for future prediction/NLP fields — no inference yet.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from src.core.case.models import ClinicalCase
from src.tasks.hemorrhage.constants import (
    EXPECTED_REPORT_TYPUS_CODES,
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)

# Stable column order for CSV export
CASE_EXPORT_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "opber_fallnr",
    "n_reports_available",
    "available_report_types",
    "missing_report_types",
    "unexpected_report_types",
    "has_operationsbericht",
    "has_eintrittsbericht",
    "has_austrittsbericht",
    "is_complete_case",
    "case_key_has_missing_component",
    "report_text_operationsbericht",
    "report_text_eintrittsbericht",
    "report_text_austrittsbericht",
    "structured_case_text",
    "construction_anomalies",
    # Prediction placeholders (Phase 1+)
    "prediction_klasse",
    "prediction_status",
    "prediction_skipped_reason",
    "llm_called",
]

_PLACEHOLDER_PREDICTION = {
    "prediction_klasse": "",
    "prediction_status": "not_run",
    "prediction_skipped_reason": "",
    "llm_called": "",
}


def _pipe_join(values: Sequence[str]) -> str:
    return "|".join(str(v) for v in values if str(v))


def case_to_export_row(case: ClinicalCase) -> dict:
    """Serialize one ``ClinicalCase`` to a flat export dict."""
    from src.core.case.keys import MISSING_KEY_TOKEN

    row = {
        "case_id": case.case_id,
        "excel_pid": case.excel_pid,
        "excel_opdat": case.excel_opdat,
        "opber_fallnr": case.opber_fallnr,
        "n_reports_available": case.n_reports_available,
        "available_report_types": _pipe_join(case.available_report_types),
        "missing_report_types": _pipe_join(case.missing_report_types),
        "unexpected_report_types": _pipe_join(case.unexpected_report_types),
        "has_operationsbericht": TYPUS_OPERATIONSBERICHT in case.reports,
        "has_eintrittsbericht": TYPUS_EINTRITTSBERICHT in case.reports,
        "has_austrittsbericht": TYPUS_AUSTRITTSBERICHT in case.reports,
        "is_complete_case": case.is_complete,
        "case_key_has_missing_component": case.case_key.has_missing_component(),
        "report_text_operationsbericht": case.get_report_text(TYPUS_OPERATIONSBERICHT),
        "report_text_eintrittsbericht": case.get_report_text(TYPUS_EINTRITTSBERICHT),
        "report_text_austrittsbericht": case.get_report_text(TYPUS_AUSTRITTSBERICHT),
        "structured_case_text": case.structured_case_text(),
        "construction_anomalies": _pipe_join(case.anomalies),
        **_PLACEHOLDER_PREDICTION,
    }
    return row


def cases_to_export_dataframe(cases: Sequence[ClinicalCase]) -> pd.DataFrame:
    """Build export DataFrame with stable column order."""
    rows = [case_to_export_row(c) for c in cases]
    if not rows:
        return pd.DataFrame(columns=CASE_EXPORT_COLUMNS)
    out = pd.DataFrame(rows)
    for col in CASE_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[CASE_EXPORT_COLUMNS]
