"""
Full structural inspection pipeline (no NLP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.pipeline.paths import (
    INSPECTION_ANOMALY_REPORT_PATH,
    INSPECTION_CASE_SUMMARY_PATH,
    INSPECTION_COLUMN_MAPPING_PATH,
    INSPECTION_DIR,
    INSPECTION_DUPLICATE_CASES_PATH,
    INSPECTION_DUPLICATE_LINKAGE_PATH,
    INSPECTION_INCOMPLETE_CASES_PATH,
    INSPECTION_KEYWORD_EXPLORATION_PATH,
    INSPECTION_MERGE_VALIDATION_PATH,
    INSPECTION_RAW_SCHEMA_PATH,
    INSPECTION_REPORT_TYPE_DISTRIBUTION_PATH,
    INSPECTION_STRUCTURED_CASE_SAMPLES_PATH,
    INSPECTION_SUMMARY_REPORT_PATH,
    INSPECTION_TEXT_FIELD_STATISTICS_PATH,
    INSPECTION_UNMATCHED_REFERENCE_PATH,
    INSPECTION_UNMATCHED_REPORTS_PATH,
)
from src.tasks.hemorrhage.config import (
    REFERENCE_XLSX_ALTERNATE_FILENAMES,
    REPORTS_XLSX_ALTERNATE_FILENAMES,
    configured_reference_xlsx_path,
    configured_reports_xlsx_path,
    reference_sheet_name,
    reports_sheet_name,
)
from src.tasks.hemorrhage.inspection.case_validation import (
    case_summary_dataframe,
    cases_per_patient_table,
    duplicate_and_anomaly_cases_dataframe,
    incomplete_cases_dataframe,
    report_type_distribution,
)
from src.tasks.hemorrhage.inspection.keyword_exploration import keyword_exploration_table
from src.tasks.hemorrhage.inspection.merge_validation import merge_validation
from src.tasks.hemorrhage.inspection.raw_schema import raw_schema_summary
from src.tasks.hemorrhage.inspection.samples import structured_case_samples
from src.tasks.hemorrhage.inspection.text_analysis import (
    case_text_length_table,
    text_field_samples,
    text_field_statistics,
)
from src.tasks.hemorrhage.constants import (
    CASE_KEY_ALIASES,
    REFERENCE_KEY_ALIASES_EXTRA,
    REFERENCE_REQUIRED_CANONICAL_KEYS,
)
from src.tasks.hemorrhage.io.column_normalize import normalize_dataframe_columns
from src.tasks.hemorrhage.io.key_normalize import merge_reference_key_aliases
from src.tasks.hemorrhage.io.excel_loader import ExcelLoadReport, load_excel_raw
from src.tasks.hemorrhage.io.path_resolve import resolve_raw_input_path
from src.tasks.hemorrhage.preprocessing.case_builder import build_cases_from_dataframe

LOGGER = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    reports_path: Path
    reference_path: Path
    reports_resolved: str
    reference_resolved: str
    reports_load: Optional[ExcelLoadReport] = None
    reference_load: Optional[ExcelLoadReport] = None
    cases_built: int = 0
    complete_cases: int = 0
    incomplete_cases: int = 0
    output_paths: List[Path] = field(default_factory=list)
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def run_full_inspection(
    *,
    reports_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> InspectionResult:
    out_dir = output_dir or INSPECTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_reports = reports_path or configured_reports_xlsx_path()
    cfg_reference = reference_path or configured_reference_xlsx_path()

    res_reports = resolve_raw_input_path(
        cfg_reports, REPORTS_XLSX_ALTERNATE_FILENAMES, context="reports"
    )
    res_reference = resolve_raw_input_path(
        cfg_reference, REFERENCE_XLSX_ALTERNATE_FILENAMES, context="reference"
    )

    result = InspectionResult(
        reports_path=res_reports.resolved_path,
        reference_path=res_reference.resolved_path,
        reports_resolved=res_reports.resolution,
        reference_resolved=res_reference.resolution,
    )

    if res_reports.resolution == "missing":
        result.errors.append(f"Reports file missing: {cfg_reports}")
    if res_reference.resolution == "missing":
        result.errors.append(f"Reference file missing: {cfg_reference}")

    schema_parts: List[pd.DataFrame] = []
    mapping_parts: List[pd.DataFrame] = []
    reports_df = pd.DataFrame()
    reference_df = pd.DataFrame()

    if res_reports.resolution != "missing":
        reports_df, result.reports_load = load_excel_raw(
            res_reports.resolved_path,
            source_label="reports_nch_export",
            sheet_name=reports_sheet_name(),
        )
        if result.reports_load.errors:
            result.errors.extend(result.reports_load.errors)
        schema_parts.append(
            raw_schema_summary(reports_df, source_label="reports", load_report=result.reports_load)
        )
        reports_df, map_rep = normalize_dataframe_columns(
            reports_df,
            source_label="reports",
            normalize_merge_keys=True,
        )
        mapping_parts.append(map_rep.to_dataframe())

    if res_reference.resolution != "missing":
        reference_df, result.reference_load = load_excel_raw(
            res_reference.resolved_path,
            source_label="reference_ccm_davf",
            sheet_name=reference_sheet_name(),
        )
        if result.reference_load.errors:
            result.errors.extend(result.reference_load.errors)
        schema_parts.append(
            raw_schema_summary(
                reference_df, source_label="reference", load_report=result.reference_load
            )
        )
        ref_aliases = merge_reference_key_aliases(CASE_KEY_ALIASES, REFERENCE_KEY_ALIASES_EXTRA)
        reference_df, map_ref = normalize_dataframe_columns(
            reference_df,
            source_label="reference",
            extra_aliases=ref_aliases,
            required_case_keys=REFERENCE_REQUIRED_CANONICAL_KEYS,
            normalize_merge_keys=True,
        )
        mapping_parts.append(map_ref.to_dataframe())

    raw_schema_df = pd.concat(schema_parts, ignore_index=True) if schema_parts else pd.DataFrame()
    mapping_df = pd.concat(mapping_parts, ignore_index=True) if mapping_parts else pd.DataFrame()

    p_raw = out_dir / INSPECTION_RAW_SCHEMA_PATH.name
    p_map = out_dir / INSPECTION_COLUMN_MAPPING_PATH.name
    _write_csv(raw_schema_df, p_raw)
    _write_csv(mapping_df, p_map)
    result.output_paths.extend([p_raw, p_map])

    cases = []
    stats = None
    if not reports_df.empty:
        cases, stats = build_cases_from_dataframe(reports_df)
        result.cases_built = len(cases)
        result.complete_cases = sum(1 for c in cases if c.is_complete)
        result.incomplete_cases = result.cases_built - result.complete_cases

    if stats is not None:
        p_case = out_dir / INSPECTION_CASE_SUMMARY_PATH.name
        _write_csv(case_summary_dataframe(cases, stats), p_case)
        result.output_paths.append(p_case)

        p_inc = out_dir / INSPECTION_INCOMPLETE_CASES_PATH.name
        _write_csv(incomplete_cases_dataframe(cases), p_inc)
        result.output_paths.append(p_inc)

        p_dup = out_dir / INSPECTION_DUPLICATE_CASES_PATH.name
        _write_csv(duplicate_and_anomaly_cases_dataframe(cases, stats), p_dup)
        result.output_paths.append(p_dup)

        p_rt = out_dir / INSPECTION_REPORT_TYPE_DISTRIBUTION_PATH.name
        _write_csv(report_type_distribution(cases, reports_df), p_rt)
        result.output_paths.append(p_rt)

        p_cpp = out_dir / "cases_per_patient.csv"
        _write_csv(cases_per_patient_table(cases), p_cpp)
        result.output_paths.append(p_cpp)

        p_samples = out_dir / INSPECTION_STRUCTURED_CASE_SAMPLES_PATH.name
        _write_csv(structured_case_samples(cases), p_samples)
        result.output_paths.append(p_samples)

    if not reports_df.empty:
        p_text = out_dir / INSPECTION_TEXT_FIELD_STATISTICS_PATH.name
        _write_csv(text_field_statistics(reports_df), p_text)
        result.output_paths.append(p_text)

        p_text_s = out_dir / "text_field_samples.csv"
        _write_csv(text_field_samples(reports_df), p_text_s)
        result.output_paths.append(p_text_s)

        p_kw = out_dir / INSPECTION_KEYWORD_EXPLORATION_PATH.name
        _write_csv(keyword_exploration_table(reports_df), p_kw)
        result.output_paths.append(p_kw)

    if not reference_df.empty and cases:
        merge_sum, un_ref, un_rep, dup_link = merge_validation(reference_df, reports_df, cases)
        p_merge = out_dir / INSPECTION_MERGE_VALIDATION_PATH.name
        _write_csv(merge_sum, p_merge)
        result.output_paths.append(p_merge)

        p_ur = out_dir / INSPECTION_UNMATCHED_REFERENCE_PATH.name
        _write_csv(un_ref.drop(columns=["_link_key"], errors="ignore"), p_ur)
        result.output_paths.append(p_ur)

        p_up = out_dir / INSPECTION_UNMATCHED_REPORTS_PATH.name
        _write_csv(un_rep.drop(columns=["_link_key"], errors="ignore"), p_up)
        result.output_paths.append(p_up)

        p_dl = out_dir / INSPECTION_DUPLICATE_LINKAGE_PATH.name
        _write_csv(dup_link, p_dl)
        result.output_paths.append(p_dl)
    elif not reference_df.empty and reports_df.empty:
        result.errors.append("reference_loaded_but_reports_empty")
    elif reference_df.empty and not reports_df.empty:
        result.errors.append("reports_loaded_but_reference_empty")

    # Detected columns summary
    detected_rows = []
    if result.reports_load:
        detected_rows.append(
            {
                "file": "reports",
                "path": str(result.reports_path),
                "columns": "|".join(result.reports_load.columns),
            }
        )
    if result.reference_load:
        detected_rows.append(
            {
                "file": "reference",
                "path": str(result.reference_path),
                "columns": "|".join(result.reference_load.columns),
            }
        )
    p_det = out_dir / "detected_columns.csv"
    _write_csv(pd.DataFrame(detected_rows), p_det)
    result.output_paths.append(p_det)

    result.summary_lines = _build_summary_lines(result, mapping_df, reports_df, reference_df)
    p_sum = out_dir / INSPECTION_SUMMARY_REPORT_PATH.name
    p_sum.write_text("\n".join(result.summary_lines) + "\n", encoding="utf-8")
    result.output_paths.append(p_sum)

    anomaly_lines = list(result.errors)
    if stats and stats.anomaly_messages:
        anomaly_lines.extend(stats.anomaly_messages[:200])
    p_anom = out_dir / INSPECTION_ANOMALY_REPORT_PATH.name
    p_anom.write_text("\n".join(anomaly_lines) + "\n", encoding="utf-8")
    result.output_paths.append(p_anom)

    return result


def _build_summary_lines(
    result: InspectionResult,
    mapping_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> List[str]:
    lines = [
        "Hemorrhage data inspection summary",
        "=" * 40,
        f"reports_path={result.reports_path} ({result.reports_resolved})",
        f"reference_path={result.reference_path} ({result.reference_resolved})",
    ]
    if result.reports_load:
        lines.append(result.reports_load.summary_line())
    if result.reference_load:
        lines.append(result.reference_load.summary_line())

    if not mapping_df.empty:
        missing = mapping_df[mapping_df["status"] == "missing"]["canonical_column"].tolist()
        if missing:
            lines.append(f"Missing canonical columns (any file): {sorted(set(missing))}")

    lines.extend(
        [
            "",
            f"cases_built={result.cases_built}",
            f"complete_cases={result.complete_cases}",
            f"incomplete_cases={result.incomplete_cases}",
            f"report_rows={len(reports_df)}",
            f"reference_rows={len(reference_df)}",
        ]
    )

    if result.errors:
        lines.append("")
        lines.append("Errors / warnings:")
        for e in result.errors:
            lines.append(f"  - {e}")

    lines.append("")
    lines.append(f"Outputs written to: {INSPECTION_DIR}")
    return lines
