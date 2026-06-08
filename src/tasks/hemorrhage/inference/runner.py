"""Case-level hemorrhage inference runner (one case = one prediction)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.core.case.models import ClinicalCase
from src.pipeline.paths import (
    HEMORRHAGE_CASE_PREDICTIONS_PATH,
    HEMORRHAGE_PARSE_FAILURES_PATH,
)
from src.tasks.hemorrhage.constants import (
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.io.load_cases import load_clinical_cases
from src.tasks.hemorrhage.io.reference_lookup import (
    LABELED_BINARY_STATUSES,
    ReferenceLookup,
    build_reference_lookup,
    reference_binary_status,
    reference_fields_for_case,
    resolve_reference_path,
)
from src.tasks.hemorrhage.inference.parse import (
    evidenz_to_json,
    list_to_json,
    parse_binary_response,
    parse_subtype_response,
    preview_snippet,
)
from src.tasks.hemorrhage.inference.prompt import (
    build_binary_messages,
    build_subtype_messages,
    prompt_preview,
)

SUBTYPE_UNCERTAIN_NOTE = "haemorrhage_subtype fehlt oder unklar (auf 'unbekannt' gesetzt)"

LOGGER = logging.getLogger(__name__)

PARSE_FAILURES_CSV_COLUMNS: List[str] = [
    "case_id",
    "raw_llm_response",
    "parse_error_reason",
    "parse_error_detail",
    "parse_repair_applied",
    "first_500_chars",
    "last_500_chars",
]

PREDICTION_CSV_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "opber_fallnr",
    "available_report_types",
    "missing_report_types",
    "has_operationsbericht",
    "has_eintrittsbericht",
    "has_austrittsbericht",
    "structured_case_text_length",
    "prompt_length_chars",
    "binary_prompt_length",
    "subtype_prompt_length",
    "status",
    "binary_stage_status",
    "subtype_stage_status",
    "klasse",
    "label",
    "haemorrhage_subtype",
    "sicherheit",
    "begruendung",
    "evidenz_json",
    "historische_blutung_erwaehnt",
    "historische_blutung_als_aktuell_gewertet",
    "unsicherheitsgruende_json",
    "raw_llm_response",
    "raw_response_length",
    "prompt_preview",
    "error_message",
    "parse_error_reason",
    "parse_error_detail",
    "parse_repair_applied",
    "reference_haemorrhagisch",
    "reference_nicht_haemorrhagisch",
    "reference_verify_vaskulaer",
    "reference_label_status",
]


@dataclass
class CasePipelineResult:
    output_path: Path
    cases_processed: int = 0
    success_count: int = 0
    dry_run_count: int = 0
    llm_failed_count: int = 0
    parse_failed_count: int = 0
    cohort_mode: str = "labeled_binary"
    cases_excluded: int = 0
    excluded_by_status: Dict[str, int] = field(default_factory=dict)
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _filter_to_cohort(
    cases: Sequence[ClinicalCase],
    ref_lookup: ReferenceLookup,
    *,
    include_verify_only: bool,
) -> "tuple[List[ClinicalCase], Dict[str, int]]":
    """
    Restrict cases to the binary-labeled evaluation cohort.

    Keeps only reference statuses ``hemorrhagic`` / ``non_hemorrhagic``. With
    ``include_verify_only`` also keep ``verify_only``. ``verify_only`` /
    ``unknown`` / ``inconsistent`` are excluded otherwise. Returns
    (kept_cases, excluded_status_counts).
    """
    allowed = set(LABELED_BINARY_STATUSES)
    if include_verify_only:
        allowed.add("verify_only")

    kept: List[ClinicalCase] = []
    excluded: Dict[str, int] = {}
    for case in cases:
        ref = reference_fields_for_case(ref_lookup, case.excel_pid, case.excel_opdat)
        status = reference_binary_status(ref)
        if status in allowed:
            kept.append(case)
        else:
            excluded[status] = excluded.get(status, 0) + 1
    return kept, excluded


def _pipe_join(values: Sequence[str]) -> str:
    return "|".join(str(v) for v in values if str(v))


def _base_row(case: ClinicalCase, ref_lookup: ReferenceLookup) -> Dict[str, Any]:
    ref = reference_fields_for_case(ref_lookup, case.excel_pid, case.excel_opdat)
    stext = case.structured_case_text()
    return {
        "case_id": case.case_id,
        "excel_pid": case.excel_pid,
        "excel_opdat": case.excel_opdat,
        "opber_fallnr": case.opber_fallnr,
        "available_report_types": _pipe_join(case.available_report_types),
        "missing_report_types": _pipe_join(case.missing_report_types),
        "has_operationsbericht": TYPUS_OPERATIONSBERICHT in case.reports,
        "has_eintrittsbericht": TYPUS_EINTRITTSBERICHT in case.reports,
        "has_austrittsbericht": TYPUS_AUSTRITTSBERICHT in case.reports,
        "structured_case_text_length": len(stext),
        "prompt_length_chars": "",
        "binary_prompt_length": "",
        "subtype_prompt_length": "",
        "status": "",
        "binary_stage_status": "",
        "subtype_stage_status": "",
        "klasse": "",
        "label": "",
        "haemorrhage_subtype": "",
        "sicherheit": "",
        "begruendung": "",
        "evidenz_json": "[]",
        "historische_blutung_erwaehnt": "",
        "historische_blutung_als_aktuell_gewertet": "",
        "unsicherheitsgruende_json": "[]",
        "raw_llm_response": "",
        "raw_response_length": 0,
        "prompt_preview": "",
        "error_message": "",
        "parse_error_reason": "",
        "parse_error_detail": "",
        "parse_repair_applied": "",
        **ref,
    }


def _apply_prediction(row: Dict[str, Any], pred: Dict[str, Any]) -> None:
    row["klasse"] = pred["klasse"] if pred["klasse"] is not None else ""
    row["label"] = pred.get("label", "")
    row["haemorrhage_subtype"] = pred.get("haemorrhage_subtype") or ""
    row["sicherheit"] = pred.get("sicherheit", "")
    row["begruendung"] = pred.get("begruendung", "")
    row["evidenz_json"] = evidenz_to_json(pred.get("evidenz") or [])
    row["historische_blutung_erwaehnt"] = (
        pred["historische_blutung_erwaehnt"]
        if pred.get("historische_blutung_erwaehnt") is not None
        else ""
    )
    row["historische_blutung_als_aktuell_gewertet"] = (
        pred["historische_blutung_als_aktuell_gewertet"]
        if pred.get("historische_blutung_als_aktuell_gewertet") is not None
        else ""
    )
    row["unsicherheitsgruende_json"] = list_to_json(pred.get("unsicherheitsgruende") or [])


def _prompt_length_chars(messages: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages or []:
        if isinstance(msg, dict):
            total += len(str(msg.get("content", "") or ""))
    return total


def _merge_subtype(
    binary_prediction: Dict[str, Any],
    subtype_result,
) -> Dict[str, Any]:
    """Combine Stage 1 (binary) prediction with Stage 2 (subtype) result."""
    merged = dict(binary_prediction)
    merged["haemorrhage_subtype"] = subtype_result.haemorrhage_subtype

    base_begr = str(merged.get("begruendung") or "").strip()
    sub_begr = str(subtype_result.begruendung or "").strip()
    if sub_begr:
        merged["begruendung"] = (
            f"{base_begr} | Subtyp: {sub_begr}" if base_begr else f"Subtyp: {sub_begr}"
        )

    merged["evidenz"] = (merged.get("evidenz") or []) + (subtype_result.evidenz or [])

    reasons = list(merged.get("unsicherheitsgruende") or [])
    for note in subtype_result.unsicherheitsgruende or []:
        if note and note not in reasons:
            reasons.append(note)
    if subtype_result.subtype_uncertain and SUBTYPE_UNCERTAIN_NOTE not in reasons:
        reasons.append(SUBTYPE_UNCERTAIN_NOTE)
    merged["unsicherheitsgruende"] = reasons
    return merged


def process_single_case(
    case: ClinicalCase,
    ref_lookup: ReferenceLookup,
    *,
    dry_run: bool = False,
    llm_call: Optional[Callable[[list], str]] = None,
    case_index: Optional[int] = None,
    total_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Two-stage hierarchical inference for one case.

    Stage 1 (binary): hämorrhagisch vs. nicht_hämorrhagisch.
    Stage 2 (subtype): only when klasse=1 — akut / nicht_akut / historisch.
    Results are merged into a single prediction row (schema unchanged).
    """
    row = _base_row(case, ref_lookup)

    binary_messages = build_binary_messages(case)
    binary_prompt_len = _prompt_length_chars(binary_messages)
    row["binary_prompt_length"] = binary_prompt_len
    row["prompt_length_chars"] = binary_prompt_len

    idx_label = f"[{case_index}/{total_cases}] " if case_index and total_cases else ""
    text_len = row["structured_case_text_length"]
    reports = row.get("available_report_types", "")
    LOGGER.info(
        "%s%s text_length=%s binary_prompt_length=%s reports=%s",
        idx_label,
        case.case_id,
        text_len,
        binary_prompt_len,
        reports,
    )

    if dry_run:
        row["status"] = "dry_run"
        row["binary_stage_status"] = "dry_run"
        row["subtype_stage_status"] = "dry_run"
        row["prompt_preview"] = prompt_preview(case)
        return row

    if llm_call is None:
        from src.tasks.hemorrhage.inference.llm_client import call_llm

        llm_call = call_llm

    # --- Stage 1: binary classification --------------------------------------
    binary_raw = ""
    try:
        binary_raw = llm_call(binary_messages) or ""
    except Exception as exc:
        row["status"] = "llm_failed"
        row["binary_stage_status"] = "llm_failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "%sLLM_FAILED stage=binary case_id=%s text_length=%s prompt_length=%s error=%s",
            idx_label,
            case.case_id,
            text_len,
            binary_prompt_len,
            row["error_message"],
        )
        return row

    row["raw_llm_response"] = binary_raw
    row["raw_response_length"] = len(binary_raw)

    binary_result = parse_binary_response(
        binary_raw, context=f"hemorrhage_binary:{case.case_id}"
    )
    _apply_prediction(row, binary_result.prediction)
    row["parse_error_reason"] = binary_result.parse_error_reason
    row["parse_error_detail"] = binary_result.parse_error_detail
    row["parse_repair_applied"] = binary_result.parse_repair_applied

    if not binary_result.success:
        row["status"] = "parse_failed"
        row["binary_stage_status"] = "parse_failed"
        row["error_message"] = binary_result.error_message
        return row

    row["binary_stage_status"] = "success"
    klasse = binary_result.prediction.get("klasse")

    # --- Stage 2 only for hemorrhagic cases ----------------------------------
    if klasse != 1:
        row["status"] = "success"
        row["subtype_stage_status"] = "skipped"
        LOGGER.info(
            "%s%s stage1=nicht_hämorrhagisch → stage2 skipped",
            idx_label,
            case.case_id,
        )
        return row

    subtype_messages = build_subtype_messages(case)
    subtype_prompt_len = _prompt_length_chars(subtype_messages)
    row["subtype_prompt_length"] = subtype_prompt_len
    row["prompt_length_chars"] = binary_prompt_len + subtype_prompt_len

    subtype_raw = ""
    try:
        subtype_raw = llm_call(subtype_messages) or ""
    except Exception as exc:
        # Stage 1 succeeded (hemorrhagic); subtype failed → keep positive,
        # subtype falls back to 'unbekannt'. Not a hard pipeline failure.
        row["subtype_stage_status"] = "llm_failed"
        row["haemorrhage_subtype"] = "unbekannt"
        reasons = list((binary_result.prediction.get("unsicherheitsgruende") or []))
        if SUBTYPE_UNCERTAIN_NOTE not in reasons:
            reasons.append(SUBTYPE_UNCERTAIN_NOTE)
        row["unsicherheitsgruende_json"] = list_to_json(reasons)
        row["status"] = "success"
        row["error_message"] = f"subtype_stage: {type(exc).__name__}: {exc}"
        LOGGER.warning(
            "%sLLM_FAILED stage=subtype case_id=%s error=%s (kept hämorrhagisch, subtype=unbekannt)",
            idx_label,
            case.case_id,
            row["error_message"],
        )
        return row

    row["raw_llm_response"] = (
        f"{binary_raw}\n\n--- SUBTYPE STAGE ---\n\n{subtype_raw}"
    )
    row["raw_response_length"] = len(binary_raw) + len(subtype_raw)

    subtype_result = parse_subtype_response(
        subtype_raw, context=f"hemorrhage_subtype:{case.case_id}"
    )
    merged = _merge_subtype(binary_result.prediction, subtype_result)
    _apply_prediction(row, merged)

    row["status"] = "success"
    row["subtype_stage_status"] = (
        "success" if subtype_result.success else "subtype_unknown"
    )
    LOGGER.info(
        "%s%s stage1=hämorrhagisch stage2=%s subtype=%s",
        idx_label,
        case.case_id,
        row["subtype_stage_status"],
        merged.get("haemorrhage_subtype"),
    )
    return row


def run_case_inference(
    cases: Sequence[ClinicalCase],
    ref_lookup: ReferenceLookup,
    *,
    dry_run: bool = False,
    llm_call: Optional[Callable[[list], str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        rows.append(
            process_single_case(case, ref_lookup, dry_run=dry_run, llm_call=llm_call)
        )
    return rows


def write_parse_failures_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    failures = [
        {
            "case_id": r.get("case_id", ""),
            "raw_llm_response": r.get("raw_llm_response", ""),
            "parse_error_reason": r.get("parse_error_reason", ""),
            "parse_error_detail": r.get("parse_error_detail", ""),
            "parse_repair_applied": r.get("parse_repair_applied", ""),
            "first_500_chars": preview_snippet(r.get("raw_llm_response", ""), 500),
            "last_500_chars": (
                str(r.get("raw_llm_response", ""))[-500:]
                if r.get("raw_llm_response")
                else ""
            ),
        }
        for r in rows
        if r.get("status") == "parse_failed"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PARSE_FAILURES_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)


def write_predictions_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_hemorrhage_case_pipeline(
    *,
    reports_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
    case_id: Optional[str] = None,
    dry_run: bool = False,
    llm_call: Optional[Callable[[list], str]] = None,
    process_all_cases: bool = False,
    include_verify_only: bool = False,
) -> CasePipelineResult:
    out_path = output_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_mode = (
        "all_cases"
        if process_all_cases
        else ("labeled_binary+verify_only" if include_verify_only else "labeled_binary")
    )
    result = CasePipelineResult(output_path=out_path, cohort_mode=cohort_mode)

    cases, stats, reports_file, load_errors = load_clinical_cases(reports_path)
    result.errors.extend(load_errors)

    ref_path = resolve_reference_path(reference_path)
    ref_status = "found" if ref_path.exists() else "missing (optional)"

    if not cases:
        print("Hemorrhage case pipeline — startup")
        print(f"  reports_path={reports_file}")
        print(f"  reference_path={ref_path} ({ref_status})")
        print(f"  cases_loaded=0")
        print(f"  output_path={out_path}")
        print(f"  dry_run={dry_run}")
        result.errors.append("no_cases_built")
        result.summary_lines = ["No cases to process.", *result.errors]
        return result

    ref_lookup, ref_errors, ref_resolved_path = build_reference_lookup(reference_path)
    result.errors.extend(ref_errors)

    work = list(cases)
    if case_id:
        work = [c for c in work if c.case_id == case_id]
        if not work:
            print("Hemorrhage case pipeline — startup")
            print(f"  reports_path={reports_file}")
            print(f"  reference_path={ref_resolved_path} ({ref_status})")
            print(f"  cases_loaded={len(cases)}")
            print(f"  output_path={out_path}")
            print(f"  dry_run={dry_run}")
            result.errors.append(f"case_id_not_found: {case_id}")
            result.summary_lines = [f"Case {case_id!r} not found among {len(cases)} cases."]
            return result

    # Cohort filtering: default to the binary-labeled cohort only. Skipped for an
    # explicit --case-id, when --all-cases is set, or when no reference labels are
    # available (cannot determine status → process all, with a warning).
    cohort_filtered = False
    if not process_all_cases and case_id is None:
        if ref_lookup:
            work, excluded = _filter_to_cohort(
                work,
                ref_lookup,
                include_verify_only=include_verify_only,
            )
            result.excluded_by_status = excluded
            result.cases_excluded = sum(excluded.values())
            cohort_filtered = True
        else:
            result.cohort_mode = "all_cases (no reference available)"
            result.errors.append(
                "cohort_filter_skipped: reference labels unavailable; processing all cases"
            )

    if limit is not None and limit > 0:
        work = work[:limit]

    timeout_seconds = max_retries = None
    if not dry_run:
        from src.tasks.hemorrhage.inference.llm_client import (
            get_max_retries,
            get_timeout_seconds,
        )

        timeout_seconds = get_timeout_seconds()
        max_retries = get_max_retries()

    print("Hemorrhage case pipeline — startup")
    print(f"  reports_path={reports_file}")
    print(f"  reference_path={ref_resolved_path} ({ref_status})")
    print(f"  cases_loaded={len(cases)}")
    print(f"  cohort_mode={result.cohort_mode}")
    if cohort_filtered:
        print(f"  cases_excluded_by_cohort={result.cases_excluded}")
        if result.excluded_by_status:
            breakdown = ", ".join(
                f"{k}={v}" for k, v in sorted(result.excluded_by_status.items())
            )
            print(f"    excluded_by_status: {breakdown}")
    print(f"  cases_to_process={len(work)}")
    print(f"  output_path={out_path}")
    print(f"  dry_run={dry_run}")
    if not dry_run:
        print(f"  llm_timeout_seconds={timeout_seconds}")
        print(f"  llm_max_retries={max_retries}")

    total = len(work)
    rows: List[Dict[str, Any]] = []

    # Incremental write: header first, then append + flush per case so completed
    # predictions are preserved even if the run is interrupted.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        f.flush()

        for idx, case in enumerate(work, start=1):
            row = process_single_case(
                case,
                ref_lookup,
                dry_run=dry_run,
                llm_call=llm_call,
                case_index=idx,
                total_cases=total,
            )
            writer.writerow(row)
            f.flush()
            rows.append(row)

            status = row.get("status", "")
            if status == "success":
                result.success_count += 1
            elif status == "dry_run":
                result.dry_run_count += 1
            elif status == "llm_failed":
                result.llm_failed_count += 1
            elif status == "parse_failed":
                result.parse_failed_count += 1

    write_parse_failures_csv(rows, HEMORRHAGE_PARSE_FAILURES_PATH)
    result.cases_processed = len(rows)

    cohort_summary = f"cohort_mode={result.cohort_mode}"
    if result.cases_excluded:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(result.excluded_by_status.items())
        )
        cohort_summary += f" (excluded={result.cases_excluded}: {breakdown})"

    result.summary_lines = [
        "Hemorrhage case pipeline (prototype)",
        f"reports_file={reports_file}",
        f"cases_total_loaded={len(cases)}",
        cohort_summary,
        f"cases_processed={result.cases_processed}",
        f"dry_run={dry_run}",
        "--- run summary ---",
        f"successful_cases={result.success_count}",
        f"parse_failed_cases={result.parse_failed_count}",
        f"llm_failed_cases={result.llm_failed_count}",
        "-------------------",
        f"dry_run_rows={result.dry_run_count}",
        f"output={out_path}",
        f"parse_failures_debug={HEMORRHAGE_PARSE_FAILURES_PATH}",
        f"input_rows={stats.input_rows}",
        f"cases_incomplete={stats.cases_incomplete}",
    ]
    if result.errors:
        result.summary_lines.append("warnings/errors:")
        for e in result.errors:
            result.summary_lines.append(f"  - {e}")

    return result
