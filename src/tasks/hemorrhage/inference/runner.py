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
    ReferenceLookup,
    build_reference_lookup,
    reference_fields_for_case,
    resolve_reference_path,
)
from src.tasks.hemorrhage.inference.parse import (
    evidenz_to_json,
    list_to_json,
    parse_hemorrhage_response,
    preview_snippet,
)
from src.tasks.hemorrhage.inference.prompt import build_messages, prompt_preview

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
    "status",
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
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


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
        "status": "",
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


def process_single_case(
    case: ClinicalCase,
    ref_lookup: ReferenceLookup,
    *,
    dry_run: bool = False,
    llm_call: Optional[Callable[[list], str]] = None,
    case_index: Optional[int] = None,
    total_cases: Optional[int] = None,
) -> Dict[str, Any]:
    row = _base_row(case, ref_lookup)

    messages = build_messages(case)
    prompt_len = _prompt_length_chars(messages)
    row["prompt_length_chars"] = prompt_len

    idx_label = (
        f"[{case_index}/{total_cases}] " if case_index and total_cases else ""
    )
    text_len = row["structured_case_text_length"]
    reports = row.get("available_report_types", "")
    LOGGER.info(
        "%s%s text_length=%s prompt_length=%s reports=%s",
        idx_label,
        case.case_id,
        text_len,
        prompt_len,
        reports,
    )

    if dry_run:
        row["status"] = "dry_run"
        row["prompt_preview"] = prompt_preview(case)
        return row

    raw = ""
    try:
        if llm_call is None:
            from src.tasks.hemorrhage.inference.llm_client import call_llm

            llm_call = call_llm
        raw = llm_call(messages) or ""
        row["raw_llm_response"] = raw
    except Exception as exc:
        row["status"] = "llm_failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning(
            "%sLLM_FAILED case_id=%s text_length=%s prompt_length=%s error=%s",
            idx_label,
            case.case_id,
            text_len,
            prompt_len,
            row["error_message"],
        )
        return row

    parse_result = parse_hemorrhage_response(raw, context=f"hemorrhage_case:{case.case_id}")
    _apply_prediction(row, parse_result.prediction)
    row["parse_error_reason"] = parse_result.parse_error_reason
    row["parse_error_detail"] = parse_result.parse_error_detail
    row["parse_repair_applied"] = parse_result.parse_repair_applied

    if parse_result.success:
        row["status"] = "success"
    else:
        row["status"] = "parse_failed"
        row["error_message"] = parse_result.error_message

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
) -> CasePipelineResult:
    out_path = output_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = CasePipelineResult(output_path=out_path)

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

    result.summary_lines = [
        "Hemorrhage case pipeline (prototype)",
        f"reports_file={reports_file}",
        f"cases_total_loaded={len(cases)}",
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
