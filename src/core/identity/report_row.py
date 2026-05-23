"""Re-export report-row identity from preprocessing (delirium compatibility)."""

from src.preprocessing.report_identity import (
    FALLBACK_MERGE_KEYS,
    PIPELINE_BERICHT_COL,
    SOURCE_REPORT_ROW_ID_COL,
    assign_source_report_row_ids,
    attach_report_identity_columns,
    choose_prediction_merge_keys,
    compute_pipeline_bericht_id,
    row_has_report_text_blocks,
)

__all__ = [
    "SOURCE_REPORT_ROW_ID_COL",
    "PIPELINE_BERICHT_COL",
    "FALLBACK_MERGE_KEYS",
    "assign_source_report_row_ids",
    "attach_report_identity_columns",
    "compute_pipeline_bericht_id",
    "choose_prediction_merge_keys",
    "row_has_report_text_blocks",
]
