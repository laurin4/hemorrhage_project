"""
Merge case-level hemorrhage classifications into a patient/case spreadsheet.

The template (``data/raw/NCH_cavernom_eingeblutet.xlsx`` by default) holds one row
per *report* — i.e. several rows can share the same case key
``(excel_pid, excel_opdat, opber_fallnr)`` (Operations-, Eintritts-, Austrittsbericht).
The classification is produced once per case and broadcast (one-hot) onto every
report row of that case.

Output classes (one-hot ``1`` / ``0``):
    - ``hämorrhagisch akut``
    - ``hämorrhagisch nicht akut``
    - ``hämorrhagisch historisch``
    - ``nicht hämorrhagisch``

Cases the model could not classify (parse_failed / llm_failed / unknown subtype /
not present in the predictions) leave all four columns blank and get a reason in
``klassifikation_status`` — so an empty cell is never confused with a real ``0``.

The raw template is never modified; a merged copy is written under ``data/outputs/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import (
    HEMORRHAGE_CASE_PREDICTIONS_PATH,
    HEMORRHAGE_CLASSIFICATION_MERGED_PATH,
    HEMORRHAGE_CLASSIFICATION_MERGE_SUMMARY_PATH,
    HEMORRHAGE_CLASSIFICATION_UNMATCHED_PATH,
)
from src.tasks.hemorrhage.config import (
    CLASSIFICATION_TEMPLATE_XLSX_ALTERNATE_FILENAMES,
    classification_template_sheet_name,
    configured_classification_template_xlsx_path,
)
from src.tasks.hemorrhage.constants import CASE_KEY_ALIASES
from src.tasks.hemorrhage.io.excel_loader import load_excel_raw
from src.tasks.hemorrhage.io.key_normalize import (
    normalize_excel_opdat_series,
    normalize_excel_pid_series,
)
from src.tasks.hemorrhage.io.path_resolve import resolve_raw_input_path

LOGGER = logging.getLogger(__name__)

# Canonical one-hot class column headers (must match the template headers).
CLASS_HAEMORRHAGIC_ACUTE = "hämorrhagisch akut"
CLASS_HAEMORRHAGIC_NON_ACUTE = "hämorrhagisch nicht akut"
CLASS_HAEMORRHAGIC_HISTORICAL = "hämorrhagisch historisch"
CLASS_NON_HAEMORRHAGIC = "nicht hämorrhagisch"

CLASS_COLUMNS: Tuple[str, ...] = (
    CLASS_HAEMORRHAGIC_ACUTE,
    CLASS_HAEMORRHAGIC_NON_ACUTE,
    CLASS_HAEMORRHAGIC_HISTORICAL,
    CLASS_NON_HAEMORRHAGIC,
)

STATUS_COLUMN = "klassifikation_status"

# Status notes for rows that receive no one-hot marker.
STATUS_OK = "klassifiziert"
STATUS_NOT_IN_PREDICTIONS = "keine_klassifikation_vorhanden"
STATUS_SUBTYPE_UNKNOWN = "haemorrhagisch_subtyp_unbekannt"
STATUS_PREDICTION_MISSING = "prediction_missing"

# Case key columns expected both in predictions and in the template.
KEY_PID = "excel_pid"
KEY_OPDAT = "excel_opdat"
KEY_FALLNR = "opber_fallnr"

CaseKeyTuple = Tuple[str, str, str]


@dataclass
class CaseClassification:
    """Resolved one-hot target for a single case."""

    class_column: Optional[str]
    status: str


@dataclass
class ClassificationMergeResult:
    template_path: Path
    output_path: Path
    summary_path: Path
    unmatched_path: Path
    template_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    classified_rows: int = 0
    unclassified_rows: int = 0
    template_cases: int = 0
    matched_cases: int = 0
    prediction_cases: int = 0
    class_row_counts: Dict[str, int] = field(default_factory=dict)
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def resolve_classification_template_path(template_path: Optional[Path] = None) -> Path:
    """Return resolved template Excel path (may be missing on disk)."""
    configured = template_path or configured_classification_template_xlsx_path()
    resolved = resolve_raw_input_path(
        configured,
        CLASSIFICATION_TEMPLATE_XLSX_ALTERNATE_FILENAMES,
        context="classification_template",
    )
    return resolved.resolved_path


def classify_prediction(label: object, subtype: object, status: object) -> CaseClassification:
    """
    Map one prediction row to its one-hot class column (or an unclassified status).

    Failures (parse_failed / llm_failed / non-success) and missing/unknown
    subtypes never receive a one-hot marker.
    """
    status_s = str(status or "").strip().lower()
    if status_s and status_s != "success":
        # parse_failed / llm_failed / dry_run / anything non-success
        return CaseClassification(class_column=None, status=status_s)

    label_s = str(label or "").strip().lower()
    subtype_s = str(subtype or "").strip().lower()

    if label_s in ("nicht_hämorrhagisch", "nicht_haemorrhagisch"):
        return CaseClassification(class_column=CLASS_NON_HAEMORRHAGIC, status=STATUS_OK)

    if label_s in ("hämorrhagisch", "haemorrhagisch"):
        if subtype_s == "akut":
            return CaseClassification(CLASS_HAEMORRHAGIC_ACUTE, STATUS_OK)
        if subtype_s == "nicht_akut":
            return CaseClassification(CLASS_HAEMORRHAGIC_NON_ACUTE, STATUS_OK)
        if subtype_s == "historisch":
            return CaseClassification(CLASS_HAEMORRHAGIC_HISTORICAL, STATUS_OK)
        # hemorrhagic but subtype missing / 'unbekannt' → no one-hot marker
        return CaseClassification(class_column=None, status=STATUS_SUBTYPE_UNKNOWN)

    # No usable binary label (e.g. empty prediction)
    return CaseClassification(class_column=None, status=STATUS_PREDICTION_MISSING)


def _normalized_key_columns(
    df: pd.DataFrame,
    *,
    pid_col: str,
    opdat_col: str,
    fallnr_col: str,
    source_label: str,
) -> pd.DataFrame:
    """Return a DataFrame of normalized (pid, opdat, fallnr) key strings."""
    pid = normalize_excel_pid_series(df[pid_col], source_label=f"{source_label}.{pid_col}")
    opdat, _ = normalize_excel_opdat_series(
        df[opdat_col], source_label=f"{source_label}.{opdat_col}"
    )
    fallnr = normalize_excel_pid_series(
        df[fallnr_col], source_label=f"{source_label}.{fallnr_col}"
    )
    return pd.DataFrame(
        {
            KEY_PID: list(pid),
            KEY_OPDAT: list(opdat),
            KEY_FALLNR: list(fallnr),
        },
        index=df.index,
    )


def build_case_classification_map(preds: pd.DataFrame) -> Dict[CaseKeyTuple, CaseClassification]:
    """
    Build ``case_key → CaseClassification`` from the predictions DataFrame.

    Keys are normalized identically to the template so the broadcast match works.
    """
    if preds.empty:
        return {}

    missing = [c for c in (KEY_PID, KEY_OPDAT, KEY_FALLNR) if c not in preds.columns]
    if missing:
        raise ValueError(f"predictions missing key columns: {missing}")

    keys = _normalized_key_columns(
        preds,
        pid_col=KEY_PID,
        opdat_col=KEY_OPDAT,
        fallnr_col=KEY_FALLNR,
        source_label="predictions",
    )

    mapping: Dict[CaseKeyTuple, CaseClassification] = {}
    for idx, row in preds.iterrows():
        key = (
            keys.at[idx, KEY_PID],
            keys.at[idx, KEY_OPDAT],
            keys.at[idx, KEY_FALLNR],
        )
        mapping[key] = classify_prediction(
            row.get("label"), row.get("haemorrhage_subtype"), row.get("status")
        )
    return mapping


def _resolve_template_key_columns(columns: List[str]) -> Dict[str, str]:
    """Resolve the actual template header for each case-key column (alias-aware)."""
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    resolved: Dict[str, str] = {}
    for canonical in (KEY_PID, KEY_OPDAT, KEY_FALLNR):
        for alias in CASE_KEY_ALIASES.get(canonical, (canonical,)):
            hit = lookup.get(alias.lower())
            if hit is not None:
                resolved[canonical] = hit
                break
    return resolved


def _resolve_class_columns(columns: List[str]) -> Dict[str, str]:
    """Map each canonical class column to the actual template header (whitespace/case-insensitive)."""

    def _norm(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    lookup = {_norm(c): str(c) for c in columns}
    resolved: Dict[str, str] = {}
    for canonical in CLASS_COLUMNS:
        hit = lookup.get(_norm(canonical))
        if hit is not None:
            resolved[canonical] = hit
    return resolved


def merge_classifications_into_template(
    template: pd.DataFrame,
    class_map: Dict[CaseKeyTuple, CaseClassification],
) -> Tuple[pd.DataFrame, ClassificationMergeResult]:
    """
    Fill one-hot class columns + status column on a copy of *template*.

    Each report row is matched on its normalized case key and receives the
    classification of its case (broadcast). Returns (merged_df, partial_result).
    Path fields on the result are placeholders; the caller fills them.
    """
    result = ClassificationMergeResult(
        template_path=Path(),
        output_path=Path(),
        summary_path=Path(),
        unmatched_path=Path(),
    )
    out = template.copy()
    result.template_rows = len(out)
    result.prediction_cases = len(class_map)

    key_cols = _resolve_template_key_columns(list(out.columns))
    missing_keys = [k for k in (KEY_PID, KEY_OPDAT, KEY_FALLNR) if k not in key_cols]
    if missing_keys:
        result.errors.append(f"template missing key columns: {missing_keys}")
        return out, result

    # Ensure class + status columns exist (create if the template lacks them).
    class_cols = _resolve_class_columns(list(out.columns))
    for canonical in CLASS_COLUMNS:
        if canonical not in class_cols:
            out[canonical] = ""
            class_cols[canonical] = canonical
            result.errors.append(f"class column not in template, created: {canonical!r}")
    if STATUS_COLUMN not in out.columns:
        out[STATUS_COLUMN] = ""

    keys = _normalized_key_columns(
        out,
        pid_col=key_cols[KEY_PID],
        opdat_col=key_cols[KEY_OPDAT],
        fallnr_col=key_cols[KEY_FALLNR],
        source_label="template",
    )

    class_row_counts = {c: 0 for c in CLASS_COLUMNS}
    template_case_keys = set()
    matched_case_keys = set()

    for idx in out.index:
        key = (
            keys.at[idx, KEY_PID],
            keys.at[idx, KEY_OPDAT],
            keys.at[idx, KEY_FALLNR],
        )
        template_case_keys.add(key)
        classification = class_map.get(key)

        if classification is None:
            for canonical in CLASS_COLUMNS:
                out.at[idx, class_cols[canonical]] = ""
            out.at[idx, STATUS_COLUMN] = STATUS_NOT_IN_PREDICTIONS
            result.unmatched_rows += 1
            continue

        matched_case_keys.add(key)
        result.matched_rows += 1

        if classification.class_column is None:
            for canonical in CLASS_COLUMNS:
                out.at[idx, class_cols[canonical]] = ""
            out.at[idx, STATUS_COLUMN] = classification.status
            result.unclassified_rows += 1
            continue

        for canonical in CLASS_COLUMNS:
            value = 1 if canonical == classification.class_column else 0
            out.at[idx, class_cols[canonical]] = value
        out.at[idx, STATUS_COLUMN] = STATUS_OK
        class_row_counts[classification.class_column] += 1
        result.classified_rows += 1

    result.template_cases = len(template_case_keys)
    result.matched_cases = len(matched_case_keys)
    result.class_row_counts = class_row_counts

    out = _reorder_columns(out, class_cols)
    return out, result


def _reorder_columns(df: pd.DataFrame, class_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Put patient-record columns first, then the one-hot class columns, then status.

    Original record-column order is preserved; the class columns follow in the
    canonical order (akut, nicht akut, historisch, nicht hämorrhagisch) and
    ``klassifikation_status`` is forced to the very end.
    """
    class_actual = [class_cols[c] for c in CLASS_COLUMNS if c in class_cols]
    appended = set(class_actual) | {STATUS_COLUMN}
    record_cols = [c for c in df.columns if c not in appended]
    ordered = record_cols + class_actual + [STATUS_COLUMN]
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]


def _unmatched_rows_frame(merged: pd.DataFrame, key_cols: Dict[str, str]) -> pd.DataFrame:
    mask = merged[STATUS_COLUMN] == STATUS_NOT_IN_PREDICTIONS
    cols = [key_cols[KEY_PID], key_cols[KEY_OPDAT], key_cols[KEY_FALLNR]]
    cols = [c for c in cols if c in merged.columns]
    return merged.loc[mask, cols].drop_duplicates().reset_index(drop=True)


def run_merge_classifications(
    *,
    predictions_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    unmatched_path: Optional[Path] = None,
) -> ClassificationMergeResult:
    """Load predictions + template, fill one-hot classes, write merged Excel."""
    pred_path = predictions_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    tmpl_path = resolve_classification_template_path(template_path)
    out_path = output_path or HEMORRHAGE_CLASSIFICATION_MERGED_PATH
    sum_path = summary_path or HEMORRHAGE_CLASSIFICATION_MERGE_SUMMARY_PATH
    unm_path = unmatched_path or HEMORRHAGE_CLASSIFICATION_UNMATCHED_PATH

    result = ClassificationMergeResult(
        template_path=tmpl_path,
        output_path=out_path,
        summary_path=sum_path,
        unmatched_path=unm_path,
    )

    if not pred_path.exists():
        result.errors.append(f"Predictions file missing: {pred_path}")
        result.summary_lines = ["No predictions to merge.", *result.errors]
        return result

    preds = pd.read_csv(pred_path, dtype=str).fillna("")
    if preds.empty:
        result.errors.append("Predictions CSV is empty")
        result.summary_lines = ["No predictions to merge.", *result.errors]
        return result

    template, load_report = load_excel_raw(
        tmpl_path,
        source_label="classification_template",
        sheet_name=classification_template_sheet_name(),
    )
    result.errors.extend(load_report.errors)
    if template.empty:
        result.errors.append(f"Template empty or missing: {tmpl_path}")
        result.summary_lines = ["No template rows to merge.", *result.errors]
        return result

    class_map = build_case_classification_map(preds)
    merged, partial = merge_classifications_into_template(template, class_map)

    # Carry over computed counts onto the full result (keep resolved paths).
    for attr in (
        "template_rows",
        "matched_rows",
        "unmatched_rows",
        "classified_rows",
        "unclassified_rows",
        "template_cases",
        "matched_cases",
        "prediction_cases",
        "class_row_counts",
    ):
        setattr(result, attr, getattr(partial, attr))
    result.errors.extend(partial.errors)

    key_cols = _resolve_template_key_columns(list(merged.columns))
    unmatched_df = _unmatched_rows_frame(merged, key_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_path, index=False, engine="openpyxl")
    unm_path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_df.to_csv(unm_path, index=False, encoding="utf-8")

    result.summary_lines = _build_summary_lines(result)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text("\n".join(result.summary_lines) + "\n", encoding="utf-8")
    return result


def _build_summary_lines(result: ClassificationMergeResult) -> List[str]:
    lines = [
        "Hemorrhage classification merge",
        f"template={result.template_path}",
        f"output={result.output_path}",
        "",
        f"template_rows={result.template_rows}",
        f"template_cases={result.template_cases}",
        f"prediction_cases={result.prediction_cases}",
        f"matched_cases={result.matched_cases}",
        f"matched_rows={result.matched_rows}",
        f"classified_rows={result.classified_rows}",
        f"unclassified_rows={result.unclassified_rows} "
        "(matched but no one-hot: failed / unknown subtype)",
        f"unmatched_rows={result.unmatched_rows} (no prediction for this case key)",
        "",
        "--- one-hot row counts ---",
    ]
    for canonical in CLASS_COLUMNS:
        lines.append(f"{canonical}={result.class_row_counts.get(canonical, 0)}")
    lines += [
        "--------------------------",
        f"unmatched_rows_csv={result.unmatched_path}",
        f"summary_txt={result.summary_path}",
    ]
    if result.errors:
        lines.append("warnings/errors:")
        for e in result.errors:
            lines.append(f"  - {e}")
    return lines
