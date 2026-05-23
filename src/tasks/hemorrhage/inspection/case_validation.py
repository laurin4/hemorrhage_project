"""Case construction validation summaries and anomaly exports."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.core.case.keys import MISSING_KEY_TOKEN
from src.core.case.models import CaseConstructionStats, ClinicalCase
from src.tasks.hemorrhage.constants import (
    EXPECTED_REPORT_TYPUS_CODES,
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.export.case_export_schema import case_to_export_row


def case_summary_dataframe(cases: Sequence[ClinicalCase], stats: CaseConstructionStats) -> pd.DataFrame:
    """High-level case construction metrics."""
    n = len(cases)
    complete = sum(1 for c in cases if c.is_complete)
    incomplete = n - complete
    zero_reports = sum(1 for c in cases if c.n_reports_available == 0)
    missing_fallnr = sum(1 for c in cases if c.opber_fallnr == MISSING_KEY_TOKEN)
    missing_opdat = sum(1 for c in cases if c.excel_opdat == MISSING_KEY_TOKEN)
    missing_pid = sum(1 for c in cases if c.excel_pid == MISSING_KEY_TOKEN)
    synthetic_key = sum(1 for c in cases if c.case_key.has_missing_component())

    patients = {c.excel_pid for c in cases if c.excel_pid != MISSING_KEY_TOKEN}
    cases_per_patient: Dict[str, int] = {}
    for c in cases:
        if c.excel_pid != MISSING_KEY_TOKEN:
            cases_per_patient[c.excel_pid] = cases_per_patient.get(c.excel_pid, 0) + 1
    max_cpp = max(cases_per_patient.values()) if cases_per_patient else 0
    mean_cpp = (sum(cases_per_patient.values()) / len(cases_per_patient)) if cases_per_patient else 0.0

    rows = [
        {"metric": "total_cases", "value": n},
        {"metric": "complete_cases_all_3_typus", "value": complete},
        {"metric": "incomplete_cases", "value": incomplete},
        {"metric": "cases_zero_reports", "value": zero_reports},
        {"metric": "unique_excel_pid", "value": len(patients)},
        {"metric": "max_cases_per_patient", "value": max_cpp},
        {"metric": "mean_cases_per_patient", "value": round(mean_cpp, 3)},
        {"metric": "cases_missing_opber_fallnr", "value": missing_fallnr},
        {"metric": "cases_missing_excel_opdat", "value": missing_opdat},
        {"metric": "cases_missing_excel_pid", "value": missing_pid},
        {"metric": "cases_with_any_missing_key_component", "value": synthetic_key},
        {"metric": "duplicate_typus_events", "value": stats.duplicate_typus_in_case},
        {"metric": "unexpected_typus_rows", "value": stats.unexpected_typus_rows},
        {"metric": "input_rows", "value": stats.input_rows},
    ]
    return pd.DataFrame(rows)


def incomplete_cases_dataframe(cases: Sequence[ClinicalCase]) -> pd.DataFrame:
    rows = [case_to_export_row(c) for c in cases if c.missing_report_types]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def duplicate_and_anomaly_cases_dataframe(
    cases: Sequence[ClinicalCase],
    stats: CaseConstructionStats,
) -> pd.DataFrame:
    flagged: List[ClinicalCase] = []
    for c in cases:
        if c.anomalies or c.unexpected_report_types or c.extra_reports:
            flagged.append(c)
        elif stats.duplicate_typus_in_case and any(
            "duplicate typus" in a for a in c.anomalies
        ):
            flagged.append(c)
    rows = [case_to_export_row(c) for c in flagged]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def report_type_distribution(cases: Sequence[ClinicalCase], reports_df: pd.DataFrame) -> pd.DataFrame:
    """Typus counts from raw rows + case combination patterns."""
    rows: List[dict] = []

    if "typus" in reports_df.columns:
        raw_counts = reports_df["typus"].fillna("").astype(str).value_counts()
        for typus, cnt in raw_counts.items():
            rows.append({"level": "raw_row", "category": typus, "count": int(cnt)})

    pattern_counts: Dict[str, int] = {}
    for c in cases:
        avail = set(c.available_report_types)
        if avail == {TYPUS_OPERATIONSBERICHT}:
            key = "only_OP"
        elif avail == {TYPUS_EINTRITTSBERICHT}:
            key = "only_Eintritt"
        elif avail == {TYPUS_AUSTRITTSBERICHT}:
            key = "only_Austritt"
        elif avail == {TYPUS_OPERATIONSBERICHT, TYPUS_EINTRITTSBERICHT}:
            key = "OP+Eintritt"
        elif avail == {TYPUS_OPERATIONSBERICHT, TYPUS_AUSTRITTSBERICHT}:
            key = "OP+Austritt"
        elif avail == {TYPUS_EINTRITTSBERICHT, TYPUS_AUSTRITTSBERICHT}:
            key = "Eintritt+Austritt"
        elif avail == set(EXPECTED_REPORT_TYPUS_CODES):
            key = "all_three"
        elif not avail:
            key = "no_typed_reports"
        else:
            key = "other_" + "_".join(sorted(avail))
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    for cat, cnt in sorted(pattern_counts.items()):
        rows.append({"level": "case_pattern", "category": cat, "count": cnt})

    unknown_rows = 0
    if "typus" in reports_df.columns:
        from src.tasks.hemorrhage.preprocessing.case_builder import normalize_typus_code

        for raw in reports_df["typus"]:
            code, _, expected = normalize_typus_code(raw)
            if not code or not expected:
                unknown_rows += 1
    rows.append({"level": "raw_row", "category": "unknown_or_unparsed_typus", "count": unknown_rows})

    return pd.DataFrame(rows)


def cases_per_patient_table(cases: Sequence[ClinicalCase]) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for c in cases:
        pid = c.excel_pid
        counts[pid] = counts.get(pid, 0) + 1
    rows = [{"excel_pid": k, "n_cases": v} for k, v in sorted(counts.items())]
    return pd.DataFrame(rows)
