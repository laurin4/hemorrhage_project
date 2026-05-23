"""
Flat report rows → case-level ``ClinicalCase`` structures.

Preserves all cases; missing report types are explicit; never requires all three typus slots.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.core.case.keys import MISSING_KEY_TOKEN, CaseKey
from src.core.case.models import (
    CaseConstructionStats,
    CaseReport,
    ClinicalCase,
    build_clinical_case,
)
from src.preprocessing.berichte_mapper import read_berichte_csv_robust
from src.tasks.hemorrhage.constants import (
    CASE_KEY_COLUMNS,
    EXPECTED_REPORT_TYPUS_CODES,
    SOURCE_ROW_ID_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    TYPUS_CODE_TO_LABEL,
    TYPUS_COLUMN_CANDIDATES,
)

LOGGER = logging.getLogger(__name__)

_TYpus_NORMALIZE_RE = re.compile(r"^\s*(\d{2})\b")


def _normalize_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    return s


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        hit = cols_lower.get(name.lower())
        if hit is not None:
            return str(hit)
    return None


def normalize_typus_code(raw: object) -> Tuple[str, str, bool]:
    """
    Map raw typus cell → (canonical_code, display_label, is_expected).

    Returns ("", raw, False) when typus cannot be parsed.
    """
    s = _normalize_str(raw)
    if not s:
        return "", "", False

    m = _TYpus_NORMALIZE_RE.match(s)
    if m:
        code = m.group(1)
        label = TYPUS_CODE_TO_LABEL.get(code, s)
        expected = code in EXPECTED_REPORT_TYPUS_CODES
        return code, label, expected

    lowered = s.lower()
    if "operationsbericht" in lowered or lowered.startswith("op"):
        return "01", TYPUS_CODE_TO_LABEL["01"], True
    if "eintritt" in lowered:
        return "02", TYPUS_CODE_TO_LABEL["02"], True
    if "austritt" in lowered:
        return "03", TYPUS_CODE_TO_LABEL["03"], True

    return "", s, False


def stitch_report_text(row: pd.Series, df_columns: Sequence[str]) -> str:
    """Assemble report body from known text columns (first non-empty blocks)."""
    parts: List[str] = []
    col_set = {str(c).strip().lower(): c for c in df_columns}
    for col_name, heading in TEXT_COLUMN_CANDIDATES:
        actual = col_set.get(col_name.lower())
        if actual is None:
            continue
        text = _normalize_str(row.get(actual, ""))
        if text:
            parts.append(f"{heading}\n{text}")
    return "\n\n".join(parts)


def load_flat_reports_dataframe(path: Path) -> pd.DataFrame:
    """Load semicolon-separated flat report file (robust parser)."""
    df = read_berichte_csv_robust(path, log_context="hemorrhage flat reports")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_cases_from_dataframe(
    df: pd.DataFrame,
    *,
    case_id_style: str = "readable",
) -> Tuple[List[ClinicalCase], CaseConstructionStats]:
    """
    Group *df* into ``ClinicalCase`` instances.

    - Never drops a case key group (even with zero text rows).
    - Duplicate (case, typus) rows: first wins, duplicates logged.
    - Unexpected typus: stored in ``extra_reports``, case still retained.
    """
    stats = CaseConstructionStats(input_rows=len(df))
    if df.empty:
        return [], stats

    pid_col, opdat_col, fallnr_col = CASE_KEY_COLUMNS
    for required in CASE_KEY_COLUMNS:
        if required not in df.columns:
            resolved = _resolve_column(df, (required,))
            if resolved is None:
                raise ValueError(
                    f"Flat report input missing case key column {required!r}. "
                    f"Found columns: {list(df.columns)}"
                )
            df = df.rename(columns={resolved: required})

    typus_col = _resolve_column(df, TYPUS_COLUMN_CANDIDATES)
    if typus_col is None:
        LOGGER.warning(
            "No typus column found (%s); cases will have zero typed reports.",
            TYPUS_COLUMN_CANDIDATES,
        )

    row_id_col = _resolve_column(df, SOURCE_ROW_ID_COLUMN_CANDIDATES)

    # Assign stable row ids when missing
    work = df.copy().reset_index(drop=True)
    if row_id_col is None:
        work["_source_row_id"] = "report_row_" + work.index.astype(str)
        row_id_col = "_source_row_id"
    elif row_id_col != "source_report_row_id":
        work = work.rename(columns={row_id_col: "source_report_row_id"})
        row_id_col = "source_report_row_id"
    if "source_report_row_id" not in work.columns:
        work["source_report_row_id"] = "report_row_" + work.index.astype(str)
        row_id_col = "source_report_row_id"

    # Case key tuple → list of row indices
    grouped: Dict[Tuple[str, str, str], List[int]] = {}
    for idx, row in work.iterrows():
        key = CaseKey.from_row(row, columns=CASE_KEY_COLUMNS)
        if key.has_missing_component():
            stats.rows_with_missing_key_component += 1
            stats.anomaly_messages.append(
                f"row {idx}: missing case key component "
                f"({key.excel_pid}, {key.excel_opdat}, {key.opber_fallnr})"
            )
        grouped.setdefault(key.parts(), []).append(int(idx))

    cases: List[ClinicalCase] = []
    for key_parts, row_indices in grouped.items():
        key = CaseKey(*key_parts)
        reports: Dict[str, CaseReport] = {}
        unexpected: Dict[str, CaseReport] = {}
        anomalies: List[str] = []

        for idx in row_indices:
            row = work.loc[idx]
            raw_typus = row[typus_col] if typus_col else ""
            code, label, expected = normalize_typus_code(raw_typus)
            text = stitch_report_text(row, work.columns)
            if not text.strip():
                stats.rows_without_text += 1
                anomalies.append(f"row {idx}: empty report text (typus={raw_typus!r})")
                # Still count row — case may be text-empty but structurally present

            meta = {
                "excel_pid": key.excel_pid,
                "excel_opdat": key.excel_opdat,
                "opber_fallnr": key.opber_fallnr,
            }
            rep = CaseReport(
                typus_code=code or "unknown",
                typus_label=label or _normalize_str(raw_typus) or "unknown",
                report_text=text,
                source_row_id=_normalize_str(row.get(row_id_col, "")),
                raw_typus=_normalize_str(raw_typus),
                metadata=meta,
            )

            if not code:
                if raw_typus:
                    stats.unexpected_typus_rows += 1
                    slot = f"unexpected_{len(unexpected)}"
                    unexpected[slot] = rep
                continue

            if not expected:
                stats.unexpected_typus_rows += 1
                unexpected[code] = rep
                anomalies.append(f"row {idx}: unexpected typus code {code!r}")
                continue

            if code in reports:
                stats.duplicate_typus_in_case += 1
                anomalies.append(
                    f"duplicate typus {code} in case {key.parts()}: "
                    f"keeping first, ignoring row {idx}"
                )
                continue

            reports[code] = rep

        case = build_clinical_case(
            key,
            reports,
            expected_typus_codes=EXPECTED_REPORT_TYPUS_CODES,
            unexpected=unexpected,
            anomalies=anomalies,
            case_id_style=case_id_style,
        )
        if case.n_reports_available == 0:
            stats.cases_with_zero_reports += 1
        if case.missing_report_types:
            stats.cases_incomplete += 1
        else:
            stats.cases_complete += 1
        cases.append(case)

    stats.cases_built = len(cases)
    return cases, stats
