"""Structured case text samples for manual NLP design review."""

from __future__ import annotations

import random
from typing import List, Sequence

import pandas as pd

from src.core.case.models import ClinicalCase


def _sample_row(case: ClinicalCase, sample_reason: str, preview_chars: int = 4000) -> dict:
    text = case.structured_case_text()
    return {
        "sample_reason": sample_reason,
        "case_id": case.case_id,
        "excel_pid": case.excel_pid,
        "excel_opdat": case.excel_opdat,
        "opber_fallnr": case.opber_fallnr,
        "n_reports_available": case.n_reports_available,
        "available_report_types": "|".join(case.available_report_types),
        "missing_report_types": "|".join(case.missing_report_types),
        "structured_case_text_length": len(text),
        "structured_case_text_preview": text[:preview_chars],
        "construction_anomalies": "|".join(case.anomalies[:10]),
    }


def structured_case_samples(
    cases: Sequence[ClinicalCase],
    *,
    n_each: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """20 random, 20 longest, 20 incomplete, 20 anomalous cases."""
    if not cases:
        return pd.DataFrame()

    rng = random.Random(seed)
    case_list = list(cases)
    rows: List[dict] = []

    # Random
    random_pick = case_list if len(case_list) <= n_each else rng.sample(case_list, n_each)
    for c in random_pick:
        rows.append(_sample_row(c, "random"))

    # Longest structured text
    by_len = sorted(case_list, key=lambda c: len(c.structured_case_text()), reverse=True)
    for c in by_len[:n_each]:
        rows.append(_sample_row(c, "longest_text"))

    # Incomplete
    incomplete = [c for c in case_list if c.missing_report_types]
    for c in incomplete[:n_each]:
        rows.append(_sample_row(c, "incomplete"))

    # Anomalous
    anomalous = [c for c in case_list if c.anomalies or c.unexpected_report_types or c.extra_reports]
    for c in anomalous[:n_each]:
        rows.append(_sample_row(c, "anomalous"))

    return pd.DataFrame(rows)
