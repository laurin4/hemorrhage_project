"""
Build unified hemorrhage prediction review table for qualitative expert inspection.

Combines model predictions, reference labels, reasoning, evidence, and compact previews.
This is preliminary comparison / qualitative review — not final evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.core.case.models import ClinicalCase
from src.pipeline.paths import (
    HEMORRHAGE_CASE_PREDICTIONS_PATH,
    HEMORRHAGE_CLINICALLY_RELEVANT_CASES_PATH,
    HEMORRHAGE_CONFUSION_REVIEW_PATH,
    HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH,
    HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH,
    HEMORRHAGE_FINAL_TARGET_SUMMARY_PATH,
    HEMORRHAGE_HISTORICAL_CASES_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH,
)
from src.tasks.hemorrhage.constants import (
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.io.load_cases import load_clinical_cases
from src.tasks.hemorrhage.io.reference_lookup import (
    build_reference_lookup,
    reference_binary_status,
    reference_fields_for_case,
)

LOGGER = logging.getLogger(__name__)

PREVIEW_MAX_CHARS = 300

REVIEW_CSV_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "opber_fallnr",
    "reference_haemorrhagisch",
    "reference_nicht_haemorrhagisch",
    "reference_verify_vaskulaer",
    "reference_status",
    "reference_haemorrhage_subtype",
    "status",
    "parse_error_reason",
    "parse_error_detail",
    "parse_repair_applied",
    "klasse",
    "label",
    "predicted_haemorrhage_subtype",
    "sicherheit",
    "historische_blutung_erwaehnt",
    "historische_blutung_als_aktuell_gewertet",
    "begruendung",
    "evidence_summary",
    "op_preview",
    "eintritt_preview",
    "austritt_preview",
    "prediction_vs_reference",
    "high_risk_mismatch",
    "needs_manual_review",
]

DETAILED_ERROR_REVIEW_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "opber_fallnr",
    "reference_status",
    "reference_haemorrhage_subtype",
    "reference_haemorrhagisch",
    "reference_nicht_haemorrhagisch",
    "reference_verify_vaskulaer",
    "status",
    "klasse",
    "label",
    "predicted_haemorrhage_subtype",
    "sicherheit",
    "prediction_vs_reference",
    "error_type",
    "begruendung",
    "evidence_summary",
    "evidenz_json",
    "historische_blutung_erwaehnt",
    "historische_blutung_als_aktuell_gewertet",
    "unsicherheitsgruende_json",
    "op_preview",
    "eintritt_preview",
    "austritt_preview",
    "raw_llm_response",
    "parse_error_reason",
    "parse_error_detail",
    "parse_repair_applied",
    "high_risk_mismatch",
    "needs_manual_review",
]

REFERENCE_STATUS_VALUES = frozenset(
    {"hemorrhagic", "non_hemorrhagic", "verify_only", "inconsistent", "unknown"}
)

PVR_VALUES = frozenset({"TP", "TN", "FP", "FN", "reference_unknown", "prediction_missing"})

CONFUSION_CSV_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "reference_status",
    "reference_haemorrhage_subtype",
    "reference_haemorrhagisch",
    "reference_nicht_haemorrhagisch",
    "reference_verify_vaskulaer",
    "status",
    "parse_error_reason",
    "parse_error_detail",
    "parse_repair_applied",
    "klasse",
    "label",
    "predicted_haemorrhage_subtype",
    "sicherheit",
    "prediction_vs_reference",
    "error_type",
    "high_risk_mismatch",
    "needs_manual_review",
]

ERROR_TYPE_BY_PVR: Dict[str, str] = {
    "TP": "correct_positive",
    "TN": "correct_negative",
    "FP": "false_positive",
    "FN": "false_negative",
    "reference_unknown": "unknown_reference",
    "prediction_missing": "pipeline_failure",
}

CONFUSION_SORT_ORDER: Dict[str, int] = {
    "FN": 0,
    "FP": 1,
    "prediction_missing": 2,
    "reference_unknown": 3,
    "TP": 4,
    "TN": 5,
}

# Final clinical target labels (derived from binary label + predicted subtype).
FINAL_TARGET_CLINICALLY_RELEVANT = "clinically_relevant_hemorrhage"
FINAL_TARGET_HISTORICAL = "historical_hemorrhage"
FINAL_TARGET_NON_HEMORRHAGIC = "non_hemorrhagic"
FINAL_TARGET_PREDICTION_MISSING = "prediction_missing"
FINAL_TARGET_PARSE_FAILED = "parse_failed"
FINAL_TARGET_LLM_FAILED = "llm_failed"

# Order used in the final-target summary CSV.
FINAL_TARGET_SUMMARY_METRICS: List[str] = [
    "total_processed_cases",
    FINAL_TARGET_CLINICALLY_RELEVANT,
    FINAL_TARGET_HISTORICAL,
    FINAL_TARGET_NON_HEMORRHAGIC,
    FINAL_TARGET_PREDICTION_MISSING,
    FINAL_TARGET_PARSE_FAILED,
    FINAL_TARGET_LLM_FAILED,
]

FINAL_TARGET_SUMMARY_COLUMNS: List[str] = ["metric", "count"]

# Full review columns + the derived final_target_label, for the split exports.
FINAL_TARGET_REVIEW_COLUMNS: List[str] = REVIEW_CSV_COLUMNS + ["final_target_label"]


@dataclass
class PredictionReviewResult:
    review_path: Path
    confusion_path: Path
    summary_path: Path
    false_negative_path: Path
    false_positive_path: Path
    clinically_relevant_path: Optional[Path] = None
    historical_path: Optional[Path] = None
    final_target_summary_path: Optional[Path] = None
    rows_written: int = 0
    fn_count: int = 0
    fp_count: int = 0
    clinically_relevant_count: int = 0
    historical_count: int = 0
    final_target_counts: Dict[str, int] = field(default_factory=dict)
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _preview_text(text: str, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    compact = " ".join(str(text or "").split())
    if not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def derive_reference_status(
    haemorrhagisch_raw: object,
    nicht_haemorrhagisch_raw: object,
    verify_vaskulaer_raw: object,
) -> str:
    """
    Derive reference_status from raw spreadsheet label cells.

    - Hämorrhagisch only → hemorrhagic
    - Nicht Hämorrhagisch only → non_hemorrhagic
    - only Verify_Vaskulär → verify_only
    - both Hämo + Nicht Hämo → inconsistent
    - everything else → unknown
    """
    return reference_binary_status(
        {
            "reference_haemorrhagisch": haemorrhagisch_raw,
            "reference_nicht_haemorrhagisch": nicht_haemorrhagisch_raw,
            "reference_verify_vaskulaer": verify_vaskulaer_raw,
        }
    )


def _parse_bool_cell(value: object) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if not s or s in ("nan", "none", "<na>"):
        return None
    if s in ("true", "1", "yes", "ja"):
        return True
    if s in ("false", "0", "no", "nein"):
        return False
    return None


def _parse_klasse(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        k = int(float(s))
        return k if k in (0, 1) else None
    except (TypeError, ValueError):
        return None


def _prediction_is_hemorrhagic(klasse: Optional[int], label: str) -> Optional[bool]:
    if klasse == 1:
        return True
    if klasse == 0:
        return False
    lab = str(label or "").strip().lower()
    if lab in ("hämorrhagisch", "haemorrhagisch"):
        return True
    if lab in ("nicht_hämorrhagisch", "nicht_haemorrhagisch", "nicht_haemorrhagisch"):
        return False
    return None


def compute_prediction_vs_reference(
    reference_status: str,
    pred_status: str,
    klasse: Optional[int],
    label: str,
) -> str:
    """TP/TN/FP/FN only for hemorrhagic/non_hemorrhagic reference labels."""
    if reference_status not in ("hemorrhagic", "non_hemorrhagic"):
        return "reference_unknown"

    if pred_status in ("parse_failed", "llm_failed", "dry_run") or not str(pred_status).strip():
        return "prediction_missing"

    pred_is = _prediction_is_hemorrhagic(klasse, label)
    if pred_is is None:
        return "prediction_missing"

    if reference_status == "hemorrhagic":
        return "TP" if pred_is else "FN"
    return "FP" if pred_is else "TN"


def compute_high_risk_mismatch(
    prediction_vs_reference: str,
    pred_status: str,
) -> bool:
    return prediction_vs_reference in ("FP", "FN") or pred_status in ("parse_failed", "llm_failed")


def _historical_bleeding_ambiguity(
    hist_mentioned: Optional[bool],
    hist_as_current: Optional[bool],
) -> bool:
    return hist_mentioned is True


def compute_needs_manual_review(
    reference_status: str,
    prediction_vs_reference: str,
    sicherheit: str,
    hist_mentioned: Optional[bool],
    hist_as_current: Optional[bool],
) -> bool:
    if reference_status in ("verify_only", "inconsistent", "unknown"):
        return True
    if str(sicherheit or "").strip().lower() == "niedrig":
        return True
    if _historical_bleeding_ambiguity(hist_mentioned, hist_as_current):
        return True
    if prediction_vs_reference in ("FP", "FN", "reference_unknown", "prediction_missing"):
        return True
    return False


def flatten_evidence_summary(evidenz_json: object) -> str:
    """Compact readable evidence string from JSON column."""
    if evidenz_json is None or (isinstance(evidenz_json, float) and pd.isna(evidenz_json)):
        return ""
    raw = str(evidenz_json).strip()
    if not raw or raw in ("[]", "nan"):
        return ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _preview_text(raw, 500)

    if not isinstance(parsed, list):
        return _preview_text(raw, 500)

    lines: List[str] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        berichttyp = str(entry.get("berichttyp", "") or "").strip()
        feld = str(entry.get("feld", "") or "").strip()
        textstelle = str(entry.get("textstelle", "") or "").strip()
        interpretation = str(entry.get("interpretation", "") or "").strip()

        prefix = "/".join(p for p in (berichttyp, feld) if p) or "evidenz"
        snippet = textstelle or interpretation
        if not snippet:
            continue
        line = f"{prefix}: {snippet}"
        if interpretation and interpretation != snippet:
            line += f" ({interpretation})"
        lines.append(line)

    return " | ".join(_preview_text(line, 200) for line in lines)


def _case_previews(case: Optional[ClinicalCase]) -> Dict[str, str]:
    if case is None:
        return {"op_preview": "", "eintritt_preview": "", "austritt_preview": ""}
    return {
        "op_preview": _preview_text(case.get_report_text(TYPUS_OPERATIONSBERICHT)),
        "eintritt_preview": _preview_text(case.get_report_text(TYPUS_EINTRITTSBERICHT)),
        "austritt_preview": _preview_text(case.get_report_text(TYPUS_AUSTRITTSBERICHT)),
    }


def derive_error_type(prediction_vs_reference: str) -> str:
    return ERROR_TYPE_BY_PVR.get(
        str(prediction_vs_reference or "").strip(),
        "unknown_reference",
    )


def build_confusion_row(review_row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact confusion-matrix style row (no long text fields)."""
    pvr = str(review_row.get("prediction_vs_reference", "") or "").strip()
    return {
        "case_id": review_row.get("case_id", ""),
        "excel_pid": review_row.get("excel_pid", ""),
        "excel_opdat": review_row.get("excel_opdat", ""),
        "reference_status": review_row.get("reference_status", ""),
        "reference_haemorrhage_subtype": review_row.get("reference_haemorrhage_subtype", ""),
        "reference_haemorrhagisch": review_row.get("reference_haemorrhagisch", ""),
        "reference_nicht_haemorrhagisch": review_row.get("reference_nicht_haemorrhagisch", ""),
        "reference_verify_vaskulaer": review_row.get("reference_verify_vaskulaer", ""),
        "status": review_row.get("status", ""),
        "parse_error_reason": review_row.get("parse_error_reason", ""),
        "parse_error_detail": review_row.get("parse_error_detail", ""),
        "parse_repair_applied": review_row.get("parse_repair_applied", ""),
        "klasse": review_row.get("klasse", ""),
        "label": review_row.get("label", ""),
        "predicted_haemorrhage_subtype": review_row.get("predicted_haemorrhage_subtype", ""),
        "sicherheit": review_row.get("sicherheit", ""),
        "prediction_vs_reference": pvr,
        "error_type": derive_error_type(pvr),
        "high_risk_mismatch": review_row.get("high_risk_mismatch", False),
        "needs_manual_review": review_row.get("needs_manual_review", False),
    }


def _confusion_sort_priority(row: Dict[str, Any]) -> Tuple[int, str]:
    pvr = str(row.get("prediction_vs_reference", "") or "").strip()
    return CONFUSION_SORT_ORDER.get(pvr, 9), str(row.get("case_id", ""))


def _sort_priority(row: Dict[str, Any]) -> Tuple[int, str]:
    pvr = str(row.get("prediction_vs_reference", ""))
    ref_status = str(row.get("reference_status", ""))
    high_risk = bool(row.get("high_risk_mismatch"))

    if pvr in ("FP", "FN"):
        tier = 0
    elif high_risk:
        tier = 1
    elif ref_status == "verify_only":
        tier = 2
    elif pvr in ("TP", "TN"):
        tier = 4
    else:
        tier = 3
    return tier, str(row.get("case_id", ""))


def _resolve_reference_fields(
    row: pd.Series,
    ref_lookup: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, str]:
    pid = str(row.get("excel_pid", "") or "").strip()
    opdat = str(row.get("excel_opdat", "") or "").strip()
    ref = reference_fields_for_case(ref_lookup, pid, opdat)

    for key in (
        "reference_haemorrhagisch",
        "reference_nicht_haemorrhagisch",
        "reference_verify_vaskulaer",
    ):
        if key not in row.index:
            continue
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, float) and val == int(val):
            s = str(int(val))
        else:
            s = str(val).strip()
        if s and s.lower() not in ("nan", "none", "<na>"):
            ref[key] = s
    return ref


def build_review_row(
    pred_row: pd.Series,
    *,
    ref_lookup: Dict[Tuple[str, str], Dict[str, str]],
    case_by_id: Dict[str, ClinicalCase],
) -> Dict[str, Any]:
    ref = _resolve_reference_fields(pred_row, ref_lookup)
    reference_status = derive_reference_status(
        ref["reference_haemorrhagisch"],
        ref["reference_nicht_haemorrhagisch"],
        ref["reference_verify_vaskulaer"],
    )

    pred_status = str(pred_row.get("status", "") or "").strip()
    klasse = _parse_klasse(pred_row.get("klasse"))
    label = str(pred_row.get("label", "") or "").strip()
    predicted_subtype = str(pred_row.get("haemorrhage_subtype", "") or "").strip()
    sicherheit = str(pred_row.get("sicherheit", "") or "").strip()

    hist_mentioned = _parse_bool_cell(pred_row.get("historische_blutung_erwaehnt"))
    hist_as_current = _parse_bool_cell(pred_row.get("historische_blutung_als_aktuell_gewertet"))

    pvr = compute_prediction_vs_reference(reference_status, pred_status, klasse, label)
    high_risk = compute_high_risk_mismatch(pvr, pred_status)
    needs_review = compute_needs_manual_review(
        reference_status, pvr, sicherheit, hist_mentioned, hist_as_current
    )

    case_id = str(pred_row.get("case_id", "") or "").strip()
    previews = _case_previews(case_by_id.get(case_id))

    return {
        "case_id": case_id,
        "excel_pid": str(pred_row.get("excel_pid", "") or "").strip(),
        "excel_opdat": str(pred_row.get("excel_opdat", "") or "").strip(),
        "opber_fallnr": str(pred_row.get("opber_fallnr", "") or "").strip(),
        "reference_haemorrhagisch": ref["reference_haemorrhagisch"],
        "reference_nicht_haemorrhagisch": ref["reference_nicht_haemorrhagisch"],
        "reference_verify_vaskulaer": ref["reference_verify_vaskulaer"],
        "reference_status": reference_status,
        "reference_haemorrhage_subtype": ref.get("reference_haemorrhage_subtype", ""),
        "status": pred_status,
        "parse_error_reason": str(pred_row.get("parse_error_reason", "") or "").strip(),
        "parse_error_detail": str(pred_row.get("parse_error_detail", "") or "").strip(),
        "parse_repair_applied": str(pred_row.get("parse_repair_applied", "") or "").strip(),
        "klasse": "" if klasse is None else klasse,
        "label": label,
        "predicted_haemorrhage_subtype": predicted_subtype,
        "sicherheit": sicherheit,
        "historische_blutung_erwaehnt": "" if hist_mentioned is None else hist_mentioned,
        "historische_blutung_als_aktuell_gewertet": "" if hist_as_current is None else hist_as_current,
        "begruendung": str(pred_row.get("begruendung", "") or "").strip(),
        "evidence_summary": flatten_evidence_summary(pred_row.get("evidenz_json")),
        **previews,
        "prediction_vs_reference": pvr,
        "high_risk_mismatch": high_risk,
        "needs_manual_review": needs_review,
    }


def _cell_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def build_detailed_error_review_row(
    review_row: Dict[str, Any],
    pred_row: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Merge review fields with raw prediction columns for manual FN/FP analysis."""
    pvr = str(review_row.get("prediction_vs_reference", "") or "").strip()
    pred = pred_row if pred_row is not None else pd.Series(dtype=object)

    return {
        "case_id": review_row.get("case_id", ""),
        "excel_pid": review_row.get("excel_pid", ""),
        "excel_opdat": review_row.get("excel_opdat", ""),
        "opber_fallnr": review_row.get("opber_fallnr", ""),
        "reference_status": review_row.get("reference_status", ""),
        "reference_haemorrhage_subtype": review_row.get("reference_haemorrhage_subtype", ""),
        "reference_haemorrhagisch": review_row.get("reference_haemorrhagisch", ""),
        "reference_nicht_haemorrhagisch": review_row.get("reference_nicht_haemorrhagisch", ""),
        "reference_verify_vaskulaer": review_row.get("reference_verify_vaskulaer", ""),
        "status": review_row.get("status", ""),
        "klasse": review_row.get("klasse", ""),
        "label": review_row.get("label", ""),
        "predicted_haemorrhage_subtype": review_row.get("predicted_haemorrhage_subtype", ""),
        "sicherheit": review_row.get("sicherheit", ""),
        "prediction_vs_reference": pvr,
        "error_type": derive_error_type(pvr),
        "begruendung": review_row.get("begruendung", ""),
        "evidence_summary": review_row.get("evidence_summary", ""),
        "evidenz_json": _cell_str(pred.get("evidenz_json", "")),
        "historische_blutung_erwaehnt": review_row.get("historische_blutung_erwaehnt", ""),
        "historische_blutung_als_aktuell_gewertet": review_row.get(
            "historische_blutung_als_aktuell_gewertet", ""
        ),
        "unsicherheitsgruende_json": _cell_str(pred.get("unsicherheitsgruende_json", "")),
        "op_preview": review_row.get("op_preview", ""),
        "eintritt_preview": review_row.get("eintritt_preview", ""),
        "austritt_preview": review_row.get("austritt_preview", ""),
        "raw_llm_response": _cell_str(pred.get("raw_llm_response", "")),
        "parse_error_reason": review_row.get("parse_error_reason", ""),
        "parse_error_detail": review_row.get("parse_error_detail", ""),
        "parse_repair_applied": review_row.get("parse_repair_applied", ""),
        "high_risk_mismatch": review_row.get("high_risk_mismatch", False),
        "needs_manual_review": review_row.get("needs_manual_review", False),
    }


def write_error_review_exports(
    review_rows: Sequence[Dict[str, Any]],
    preds_df: pd.DataFrame,
    *,
    false_negative_path: Path,
    false_positive_path: Path,
) -> Tuple[int, int]:
    """
    Write detailed FN/FP review CSVs.

    Only rows with prediction_vs_reference == FN or FP (strict).
    """
    pred_by_id: Dict[str, pd.Series] = {}
    if "case_id" in preds_df.columns:
        for _, row in preds_df.iterrows():
            cid = str(row.get("case_id", "") or "").strip()
            if cid:
                pred_by_id[cid] = row

    fn_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    for review_row in review_rows:
        pvr = str(review_row.get("prediction_vs_reference", "") or "").strip()
        if pvr == "FN":
            fn_rows.append(
                build_detailed_error_review_row(
                    review_row, pred_by_id.get(str(review_row.get("case_id", "")))
                )
            )
        elif pvr == "FP":
            fp_rows.append(
                build_detailed_error_review_row(
                    review_row, pred_by_id.get(str(review_row.get("case_id", "")))
                )
            )

    false_negative_path.parent.mkdir(parents=True, exist_ok=True)
    false_positive_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fn_rows, columns=DETAILED_ERROR_REVIEW_COLUMNS).to_csv(
        false_negative_path, index=False, encoding="utf-8"
    )
    pd.DataFrame(fp_rows, columns=DETAILED_ERROR_REVIEW_COLUMNS).to_csv(
        false_positive_path, index=False, encoding="utf-8"
    )
    return len(fn_rows), len(fp_rows)


def derive_final_target_label(review_row: Dict[str, Any]) -> str:
    """
    Map a review row to its final clinical target label.

    Priority: pipeline failures first, then the binary label, then (for
    hemorrhagic) the predicted subtype. Every row maps to exactly one label, so
    the six categories partition all processed cases.
    """
    status = str(review_row.get("status", "") or "").strip().lower()
    if status == "llm_failed":
        return FINAL_TARGET_LLM_FAILED
    if status == "parse_failed":
        return FINAL_TARGET_PARSE_FAILED

    label = str(review_row.get("label", "") or "").strip().lower()
    subtype = str(review_row.get("predicted_haemorrhage_subtype", "") or "").strip().lower()

    if label == "hämorrhagisch":
        if subtype == "historisch":
            return FINAL_TARGET_HISTORICAL
        return FINAL_TARGET_CLINICALLY_RELEVANT
    if label == "nicht_hämorrhagisch":
        return FINAL_TARGET_NON_HEMORRHAGIC
    return FINAL_TARGET_PREDICTION_MISSING


def _final_target_review_row(review_row: Dict[str, Any], target_label: str) -> Dict[str, Any]:
    out = {col: review_row.get(col, "") for col in REVIEW_CSV_COLUMNS}
    out["final_target_label"] = target_label
    return out


def write_final_target_exports(
    review_rows: Sequence[Dict[str, Any]],
    *,
    clinically_relevant_path: Path,
    historical_path: Path,
) -> Tuple[int, int]:
    """
    Split hemorrhagic predictions into two manual-review CSVs.

    - clinically relevant: label == hämorrhagisch AND subtype != historisch
    - historical:          label == hämorrhagisch AND subtype == historisch

    Together these cover every hemorrhagic prediction exactly once. Returns
    (clinically_relevant_count, historical_count).
    """
    relevant_rows: List[Dict[str, Any]] = []
    historical_rows: List[Dict[str, Any]] = []
    for review_row in review_rows:
        target = derive_final_target_label(review_row)
        if target == FINAL_TARGET_CLINICALLY_RELEVANT:
            relevant_rows.append(_final_target_review_row(review_row, target))
        elif target == FINAL_TARGET_HISTORICAL:
            historical_rows.append(_final_target_review_row(review_row, target))

    clinically_relevant_path.parent.mkdir(parents=True, exist_ok=True)
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(relevant_rows, columns=FINAL_TARGET_REVIEW_COLUMNS).to_csv(
        clinically_relevant_path, index=False, encoding="utf-8"
    )
    pd.DataFrame(historical_rows, columns=FINAL_TARGET_REVIEW_COLUMNS).to_csv(
        historical_path, index=False, encoding="utf-8"
    )
    return len(relevant_rows), len(historical_rows)


def compute_final_target_counts(review_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Counts per final target label (+ total_processed_cases)."""
    counts: Dict[str, int] = {m: 0 for m in FINAL_TARGET_SUMMARY_METRICS}
    counts["total_processed_cases"] = len(review_rows)
    for review_row in review_rows:
        counts[derive_final_target_label(review_row)] += 1
    return counts


def write_final_target_summary(
    review_rows: Sequence[Dict[str, Any]],
    summary_path: Path,
) -> Dict[str, int]:
    """Write hemorrhage_final_target_summary.csv (metric,count) and return counts."""
    counts = compute_final_target_counts(review_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": m, "count": counts[m]} for m in FINAL_TARGET_SUMMARY_METRICS]
    pd.DataFrame(rows, columns=FINAL_TARGET_SUMMARY_COLUMNS).to_csv(
        summary_path, index=False, encoding="utf-8"
    )
    return counts


def build_prediction_review_summary(
    review_rows: Sequence[Dict[str, Any]],
    *,
    false_negative_path: Optional[Path] = None,
    false_positive_path: Optional[Path] = None,
    fn_count: Optional[int] = None,
    fp_count: Optional[int] = None,
) -> List[str]:
    """Preliminary labeled-subset comparison summary (not final evaluation)."""
    total = len(review_rows)
    tp = sum(1 for r in review_rows if r.get("prediction_vs_reference") == "TP")
    tn = sum(1 for r in review_rows if r.get("prediction_vs_reference") == "TN")
    fp = sum(1 for r in review_rows if r.get("prediction_vs_reference") == "FP")
    fn = sum(1 for r in review_rows if r.get("prediction_vs_reference") == "FN")
    prediction_missing = sum(
        1 for r in review_rows if r.get("prediction_vs_reference") == "prediction_missing"
    )
    reference_unknown_pvr = sum(
        1 for r in review_rows if r.get("prediction_vs_reference") == "reference_unknown"
    )
    verify_only = sum(1 for r in review_rows if r.get("reference_status") == "verify_only")
    unknown = sum(1 for r in review_rows if r.get("reference_status") == "unknown")
    inconsistent = sum(1 for r in review_rows if r.get("reference_status") == "inconsistent")
    parse_failed = sum(1 for r in review_rows if r.get("status") == "parse_failed")
    llm_failed = sum(1 for r in review_rows if r.get("status") == "llm_failed")

    labeled_compared = tp + tn + fp + fn
    mismatches = fp + fn
    mismatch_rate = (mismatches / labeled_compared) if labeled_compared else None

    lines = [
        "Hemorrhage prediction review — preliminary comparison summary",
        "(Qualitative review / labeled subset comparison — NOT final evaluation)",
        "",
        f"total_cases={total}",
        f"TP={tp}",
        f"TN={tn}",
        f"FP={fp}",
        f"FN={fn}",
        f"prediction_missing={prediction_missing}",
        f"reference_unknown={reference_unknown_pvr}",
        f"verify_only={verify_only} (excluded from performance stats)",
        f"reference_status_unknown={unknown}",
        f"reference_inconsistent={inconsistent}",
        f"parse_failed={parse_failed}",
        f"llm_failed={llm_failed}",
        f"labeled_subset_compared={labeled_compared}",
    ]
    if mismatch_rate is not None:
        lines.append(
            f"mismatch_rate_labeled_subset_only={mismatch_rate:.3f} "
            f"({mismatches}/{labeled_compared})"
        )
    else:
        lines.append("mismatch_rate_labeled_subset_only=n/a (no labeled TP/TN/FP/FN cases)")

    if fn_count is not None:
        lines.append(f"FN_count={fn_count}")
    if fp_count is not None:
        lines.append(f"FP_count={fp_count}")
    if false_negative_path is not None:
        lines.append(f"false_negative_review_path={false_negative_path}")
    if false_positive_path is not None:
        lines.append(f"false_positive_review_path={false_positive_path}")
    return lines


def run_build_prediction_review(
    *,
    predictions_path: Optional[Path] = None,
    review_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    reports_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    limit: Optional[int] = None,
    only_mismatches: bool = False,
    only_labeled: bool = False,
    only_fn: bool = False,
    only_fp: bool = False,
    false_negative_path: Optional[Path] = None,
    false_positive_path: Optional[Path] = None,
    clinically_relevant_path: Optional[Path] = None,
    historical_path: Optional[Path] = None,
    final_target_summary_path: Optional[Path] = None,
    skip_main_exports: bool = False,
) -> PredictionReviewResult:
    pred_path = predictions_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    out_path = review_path or HEMORRHAGE_PREDICTION_REVIEW_PATH
    conf_path = confusion_path or HEMORRHAGE_CONFUSION_REVIEW_PATH
    sum_path = summary_path or HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH
    fn_path = false_negative_path or HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH
    fp_path = false_positive_path or HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH
    relevant_path = clinically_relevant_path or HEMORRHAGE_CLINICALLY_RELEVANT_CASES_PATH
    hist_path = historical_path or HEMORRHAGE_HISTORICAL_CASES_PATH
    target_sum_path = final_target_summary_path or HEMORRHAGE_FINAL_TARGET_SUMMARY_PATH
    result = PredictionReviewResult(
        review_path=out_path,
        confusion_path=conf_path,
        summary_path=sum_path,
        false_negative_path=fn_path,
        false_positive_path=fp_path,
        clinically_relevant_path=relevant_path,
        historical_path=hist_path,
        final_target_summary_path=target_sum_path,
    )

    if not pred_path.exists():
        result.errors.append(f"Predictions file missing: {pred_path}")
        result.summary_lines = ["No predictions to review.", *result.errors]
        return result

    preds = pd.read_csv(pred_path)
    if preds.empty:
        result.errors.append("Predictions CSV is empty")
        result.summary_lines = ["No predictions to review.", *result.errors]
        return result

    ref_lookup, ref_errors, _ = build_reference_lookup(reference_path)
    result.errors.extend(ref_errors)

    cases, _, _, case_errors = load_clinical_cases(reports_path)
    result.errors.extend(case_errors)
    case_by_id = {c.case_id: c for c in cases}

    review_rows: List[Dict[str, Any]] = []
    for _, row in preds.iterrows():
        review_rows.append(
            build_review_row(row, ref_lookup=ref_lookup, case_by_id=case_by_id)
        )

    # Final-target exports operate on ALL processed predictions, independent of
    # the label/mismatch filters below (every hemorrhagic prediction must appear
    # in exactly one of the clinically-relevant / historical CSVs).
    full_review_rows = list(review_rows)
    clinically_relevant_count, historical_count = write_final_target_exports(
        full_review_rows,
        clinically_relevant_path=relevant_path,
        historical_path=hist_path,
    )
    final_target_counts = write_final_target_summary(full_review_rows, target_sum_path)
    result.clinically_relevant_count = clinically_relevant_count
    result.historical_count = historical_count
    result.final_target_counts = final_target_counts

    if only_labeled:
        review_rows = [
            r
            for r in review_rows
            if r["reference_status"] in ("hemorrhagic", "non_hemorrhagic")
        ]

    if only_mismatches:
        review_rows = [
            r
            for r in review_rows
            if r["prediction_vs_reference"] in ("FP", "FN")
            or r["status"] in ("parse_failed", "llm_failed")
        ]

    summary_rows = list(review_rows)

    fn_count, fp_count = write_error_review_exports(
        summary_rows,
        preds,
        false_negative_path=fn_path,
        false_positive_path=fp_path,
    )
    result.fn_count = fn_count
    result.fp_count = fp_count

    if only_fn:
        review_rows = [r for r in review_rows if r["prediction_vs_reference"] == "FN"]
    elif only_fp:
        review_rows = [r for r in review_rows if r["prediction_vs_reference"] == "FP"]

    confusion_rows = [build_confusion_row(r) for r in review_rows]
    confusion_rows.sort(key=_confusion_sort_priority)

    review_rows.sort(key=_sort_priority)

    if limit is not None and limit > 0:
        review_rows = review_rows[:limit]
        confusion_rows = confusion_rows[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_main_exports:
        pd.DataFrame(review_rows, columns=REVIEW_CSV_COLUMNS).to_csv(
            out_path, index=False, encoding="utf-8"
        )
        pd.DataFrame(confusion_rows, columns=CONFUSION_CSV_COLUMNS).to_csv(
            conf_path, index=False, encoding="utf-8"
        )
        result.rows_written = len(review_rows)
    else:
        result.rows_written = fn_count + fp_count

    summary_lines = build_prediction_review_summary(
        summary_rows,
        false_negative_path=fn_path,
        false_positive_path=fp_path,
        fn_count=fn_count,
        fp_count=fp_count,
    )
    sum_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    counts = result.final_target_counts
    hemorrhagic_total = clinically_relevant_count + historical_count
    result.summary_lines = summary_lines + [
        "",
        "--- final target summary ---",
        f"total_processed_cases={counts.get('total_processed_cases', 0)}",
        f"clinically_relevant_hemorrhage={counts.get(FINAL_TARGET_CLINICALLY_RELEVANT, 0)}",
        f"historical_hemorrhage={counts.get(FINAL_TARGET_HISTORICAL, 0)}",
        f"non_hemorrhagic={counts.get(FINAL_TARGET_NON_HEMORRHAGIC, 0)}",
        f"prediction_missing={counts.get(FINAL_TARGET_PREDICTION_MISSING, 0)}",
        f"parse_failed={counts.get(FINAL_TARGET_PARSE_FAILED, 0)}",
        f"llm_failed={counts.get(FINAL_TARGET_LLM_FAILED, 0)}",
        f"hemorrhagic_predictions_total={hemorrhagic_total} "
        f"(clinically_relevant={clinically_relevant_count} + historical={historical_count})",
        "----------------------------",
        "",
        f"review_csv={out_path}",
        f"confusion_csv={conf_path}",
        f"false_negative_review_csv={fn_path}",
        f"false_positive_review_csv={fp_path}",
        f"clinically_relevant_cases_csv={relevant_path}",
        f"historical_cases_csv={hist_path}",
        f"final_target_summary_csv={target_sum_path}",
        f"summary_txt={sum_path}",
        f"rows_written={result.rows_written}",
        f"FN_export_rows={fn_count}",
        f"FP_export_rows={fp_count}",
    ]
    if result.errors:
        result.summary_lines.append("warnings:")
        for e in result.errors:
            result.summary_lines.append(f"  - {e}")

    return result
