"""Reference ↔ report linkage validation (no label inference)."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import pandas as pd

from src.core.case.models import ClinicalCase
from src.tasks.hemorrhage.constants import CASE_KEY_COLUMNS, MERGE_LINK_KEY_COLUMNS

LOGGER = logging.getLogger(__name__)


def _link_key_tuple(row: pd.Series, columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(c, "") or "").strip() for c in columns)


def merge_validation(
    reference_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    cases: Sequence[ClinicalCase],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate linkage on ``MERGE_LINK_KEY_COLUMNS`` (excel_pid, excel_opdat).

    Reference (CCM DAVF) typically has no ``opber_fallnr``; case construction still
    uses all three keys on the reports side only.
    """
    link_cols = MERGE_LINK_KEY_COLUMNS

    case_by_link: dict[Tuple[str, ...], str] = {}
    for c in cases:
        case_by_link[(c.excel_pid.strip(), c.excel_opdat.strip())] = c.case_id

    ref = reference_df.copy()
    rep = reports_df.copy()

    for col in link_cols:
        if col not in ref.columns:
            ref[col] = ""
        if col not in rep.columns:
            rep[col] = ""

    ref["_link_key"] = ref.apply(lambda r: _link_key_tuple(r, link_cols), axis=1)
    rep["_link_key"] = rep.apply(lambda r: _link_key_tuple(r, link_cols), axis=1)

    ref_key_counts = ref["_link_key"].value_counts()
    rep_key_counts = rep["_link_key"].value_counts()

    ref_dup_keys = ref_key_counts[ref_key_counts > 1]
    rep_dup_keys = rep_key_counts[rep_key_counts > 1]

    ref_keys = set(ref["_link_key"].tolist())
    rep_keys = set(rep["_link_key"].tolist())

    matched_keys = ref_keys & rep_keys
    ref_only = ref_keys - rep_keys
    rep_only = rep_keys - ref_keys

    ref_matched_rows = int(ref["_link_key"].isin(matched_keys).sum())
    rep_matched_rows = int(rep["_link_key"].isin(matched_keys).sum())

    summary_rows: List[dict] = [
        {"metric": "merge_key_columns", "value": ",".join(link_cols)},
        {"metric": "reference_rows", "value": len(ref)},
        {"metric": "report_rows", "value": len(rep)},
        {"metric": "unique_cases_built", "value": len(cases)},
        {"metric": "unique_reference_link_keys", "value": len(ref_keys)},
        {"metric": "unique_report_link_keys", "value": len(rep_keys)},
        {"metric": "matched_link_keys", "value": len(matched_keys)},
        {"metric": "reference_rows_with_report_match", "value": ref_matched_rows},
        {"metric": "report_rows_with_reference_match", "value": rep_matched_rows},
        {"metric": "reference_keys_without_reports", "value": len(ref_only)},
        {"metric": "report_keys_without_reference", "value": len(rep_only)},
        {"metric": "reference_duplicate_link_keys", "value": len(ref_dup_keys)},
        {"metric": "report_duplicate_link_keys", "value": len(rep_dup_keys)},
    ]

    m2o = int((ref_dup_keys > 0).sum()) if len(ref_dup_keys) else 0
    summary_rows.append({"metric": "many_to_one_reference_link_keys", "value": m2o})

    summary_df = pd.DataFrame(summary_rows)

    unmatched_ref = ref[ref["_link_key"].isin(ref_only)].copy()
    unmatched_rep = rep[rep["_link_key"].isin(rep_only)].copy()

    dup_rows: List[dict] = []
    for key, cnt in ref_dup_keys.items():
        dup_rows.append(
            {
                "linkage_type": "reference_many_per_link_key",
                "excel_pid": key[0] if len(key) > 0 else "",
                "excel_opdat": key[1] if len(key) > 1 else "",
                "row_count": int(cnt),
                "case_id": case_by_link.get(key, ""),
            }
        )
    for key, cnt in rep_dup_keys.items():
        dup_rows.append(
            {
                "linkage_type": "report_many_per_link_key",
                "excel_pid": key[0] if len(key) > 0 else "",
                "excel_opdat": key[1] if len(key) > 1 else "",
                "row_count": int(cnt),
                "case_id": case_by_link.get(key, ""),
            }
        )
    duplicate_linkage_df = pd.DataFrame(dup_rows)

    if len(ref_only):
        LOGGER.warning(
            "merge_validation: %d reference link keys have no report rows (keys=%s)",
            len(ref_only),
            ",".join(link_cols),
        )
    if len(rep_only):
        LOGGER.warning(
            "merge_validation: %d report link keys have no reference rows",
            len(rep_only),
        )

    return summary_df, unmatched_ref, unmatched_rep, duplicate_linkage_df
