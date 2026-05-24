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
    HEMORRHAGE_CONFUSION_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH,
)
from src.tasks.hemorrhage.analysis.reference_labels import parse_label_value
from src.tasks.hemorrhage.constants import (
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.io.load_cases import load_clinical_cases
from src.tasks.hemorrhage.io.reference_lookup import build_reference_lookup, reference_fields_for_case

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
    "status",
    "parse_error_reason",
    "parse_error_detail",
    "klasse",
    "label",
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

REFERENCE_STATUS_VALUES = frozenset(
    {"hemorrhagic", "non_hemorrhagic", "verify_only", "inconsistent", "unknown"}
)

PVR_VALUES = frozenset({"TP", "TN", "FP", "FN", "reference_unknown", "prediction_missing"})

CONFUSION_CSV_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "reference_status",
    "reference_haemorrhagisch",
    "reference_nicht_haemorrhagisch",
    "reference_verify_vaskulaer",
    "status",
    "parse_error_reason",
    "parse_error_detail",
    "klasse",
    "label",
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


@dataclass
class PredictionReviewResult:
    review_path: Path
    confusion_path: Path
    summary_path: Path
    rows_written: int = 0
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
    h_state, _ = parse_label_value(haemorrhagisch_raw)
    n_state, _ = parse_label_value(nicht_haemorrhagisch_raw)
    v_state, _ = parse_label_value(verify_vaskulaer_raw)

    if h_state == "yes" and n_state == "yes":
        return "inconsistent"
    if h_state == "yes" and n_state != "yes":
        return "hemorrhagic"
    if n_state == "yes" and h_state != "yes":
        return "non_hemorrhagic"
    if v_state == "yes" and h_state != "yes" and n_state != "yes":
        return "verify_only"
    return "unknown"


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
        "reference_haemorrhagisch": review_row.get("reference_haemorrhagisch", ""),
        "reference_nicht_haemorrhagisch": review_row.get("reference_nicht_haemorrhagisch", ""),
        "reference_verify_vaskulaer": review_row.get("reference_verify_vaskulaer", ""),
        "status": review_row.get("status", ""),
        "parse_error_reason": review_row.get("parse_error_reason", ""),
        "parse_error_detail": review_row.get("parse_error_detail", ""),
        "klasse": review_row.get("klasse", ""),
        "label": review_row.get("label", ""),
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
        "status": pred_status,
        "parse_error_reason": str(pred_row.get("parse_error_reason", "") or "").strip(),
        "parse_error_detail": str(pred_row.get("parse_error_detail", "") or "").strip(),
        "klasse": "" if klasse is None else klasse,
        "label": label,
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


def build_prediction_review_summary(review_rows: Sequence[Dict[str, Any]]) -> List[str]:
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
) -> PredictionReviewResult:
    pred_path = predictions_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    out_path = review_path or HEMORRHAGE_PREDICTION_REVIEW_PATH
    conf_path = confusion_path or HEMORRHAGE_CONFUSION_REVIEW_PATH
    sum_path = summary_path or HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH
    result = PredictionReviewResult(
        review_path=out_path,
        confusion_path=conf_path,
        summary_path=sum_path,
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

    confusion_rows = [build_confusion_row(r) for r in review_rows]
    confusion_rows.sort(key=_confusion_sort_priority)

    review_rows.sort(key=_sort_priority)

    if limit is not None and limit > 0:
        review_rows = review_rows[:limit]
        confusion_rows = confusion_rows[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(review_rows, columns=REVIEW_CSV_COLUMNS).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    pd.DataFrame(confusion_rows, columns=CONFUSION_CSV_COLUMNS).to_csv(
        conf_path, index=False, encoding="utf-8"
    )
    result.rows_written = len(review_rows)

    summary_lines = build_prediction_review_summary(summary_rows)
    sum_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    result.summary_lines = summary_lines + [
        "",
        f"review_csv={out_path}",
        f"confusion_csv={conf_path}",
        f"summary_txt={sum_path}",
        f"rows_written={result.rows_written}",
    ]
    if result.errors:
        result.summary_lines.append("warnings:")
        for e in result.errors:
            result.summary_lines.append(f"  - {e}")

    return result
