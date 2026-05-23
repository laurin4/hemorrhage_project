"""Reference ↔ report linkage validation (no label inference)."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import pandas as pd

from src.core.case.models import ClinicalCase
from src.tasks.hemorrhage.constants import CASE_KEY_COLUMNS

LOGGER = logging.getLogger(__name__)


def _case_key_tuple(row: pd.Series) -> Tuple[str, str, str]:
    return tuple(str(row.get(c, "") or "").strip() for c in CASE_KEY_COLUMNS)


def merge_validation(
    reference_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    cases: Sequence[ClinicalCase],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate linkage on ``CASE_KEY_COLUMNS``.

    Returns:
        summary_df, unmatched_reference_df, unmatched_reports_df, duplicate_linkage_df
    """
    case_keys = {
        (c.excel_pid, c.excel_opdat, c.opber_fallnr): c.case_id for c in cases
    }
    case_key_set = set(case_keys.keys())

    ref = reference_df.copy()
    rep = reports_df.copy()

    for col in CASE_KEY_COLUMNS:
        if col not in ref.columns:
            ref[col] = ""
        if col not in rep.columns:
            rep[col] = ""

    ref["_case_key"] = ref.apply(_case_key_tuple, axis=1)
    rep["_case_key"] = rep.apply(_case_key_tuple, axis=1)

    ref_key_counts = ref["_case_key"].value_counts()
    rep_key_counts = rep["_case_key"].value_counts()

    ref_dup_keys = ref_key_counts[ref_key_counts > 1]
    rep_dup_keys = rep_key_counts[rep_key_counts > 1]

    ref_keys = set(ref["_case_key"].tolist())
    rep_keys = set(rep["_case_key"].tolist())

    matched_keys = ref_keys & rep_keys
    ref_only = ref_keys - rep_keys
    rep_only = rep_keys - ref_keys

    ref_matched_rows = int(ref["_case_key"].isin(matched_keys).sum())
    rep_matched_rows = int(rep["_case_key"].isin(matched_keys).sum())

    summary_rows: List[dict] = [
        {"metric": "reference_rows", "value": len(ref)},
        {"metric": "report_rows", "value": len(rep)},
        {"metric": "unique_cases_built", "value": len(cases)},
        {"metric": "unique_reference_case_keys", "value": len(ref_keys)},
        {"metric": "unique_report_case_keys", "value": len(rep_keys)},
        {"metric": "matched_case_keys", "value": len(matched_keys)},
        {"metric": "reference_rows_with_report_match", "value": ref_matched_rows},
        {"metric": "report_rows_with_reference_match", "value": rep_matched_rows},
        {"metric": "reference_keys_without_reports", "value": len(ref_only)},
        {"metric": "report_keys_without_reference", "value": len(rep_only)},
        {"metric": "reference_duplicate_keys", "value": len(ref_dup_keys)},
        {"metric": "report_duplicate_keys", "value": len(rep_dup_keys)},
    ]

    # many-to-one: multiple reference rows per case key
    m2o = int((ref_dup_keys > 0).sum()) if len(ref_dup_keys) else 0
    summary_rows.append({"metric": "many_to_one_reference_keys", "value": m2o})

    summary_df = pd.DataFrame(summary_rows)

    unmatched_ref = ref[ref["_case_key"].isin(ref_only)].copy()
    unmatched_rep = rep[rep["_case_key"].isin(rep_only)].copy()

    dup_rows: List[dict] = []
    for key, cnt in ref_dup_keys.items():
        dup_rows.append(
            {
                "linkage_type": "reference_many_per_case_key",
                "excel_pid": key[0],
                "excel_opdat": key[1],
                "opber_fallnr": key[2],
                "row_count": int(cnt),
                "case_id": case_keys.get(key, ""),
            }
        )
    for key, cnt in rep_dup_keys.items():
        dup_rows.append(
            {
                "linkage_type": "report_many_per_case_key",
                "excel_pid": key[0],
                "excel_opdat": key[1],
                "opber_fallnr": key[2],
                "row_count": int(cnt),
                "case_id": case_keys.get(key, ""),
            }
        )
    duplicate_linkage_df = pd.DataFrame(dup_rows)

    if len(ref_only):
        LOGGER.warning("merge_validation: %d reference case keys have no report rows", len(ref_only))
    if len(rep_only):
        LOGGER.warning("merge_validation: %d report case keys have no reference rows", len(rep_only))

    return summary_df, unmatched_ref, unmatched_rep, duplicate_linkage_df
