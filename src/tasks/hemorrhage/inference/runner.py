"""Case-level hemorrhage inference runner (one case = one prediction)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.core.case.models import ClinicalCase
from src.pipeline.paths import HEMORRHAGE_CASE_PREDICTIONS_PATH
from src.tasks.hemorrhage.constants import (
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.data.load_cases import load_clinical_cases
from src.tasks.hemorrhage.data.reference_lookup import (
    ReferenceLookup,
    build_reference_lookup,
    reference_fields_for_case,
)
from src.tasks.hemorrhage.inference.parse import (
    evidenz_to_json,
    list_to_json,
    parse_hemorrhage_response,
)
from src.tasks.hemorrhage.inference.prompt import build_messages, prompt_preview

LOGGER = logging.getLogger(__name__)

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
    "status",
    "klasse",
    "label",
    "sicherheit",
    "begruendung",
    "evidenz_json",
    "historische_blutung_erwaehnt",
    "historische_blutung_als_aktuell_gewertet",
    "unsicherheitsgruende_json",
    "raw_llm_response",
    "prompt_preview",
    "error_message",
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
        "status": "",
        "klasse": "",
        "label": "",
        "sicherheit": "",
        "begruendung": "",
        "evidenz_json": "[]",
        "historische_blutung_erwaehnt": "",
        "historische_blutung_als_aktuell_gewertet": "",
        "unsicherheitsgruende_json": "[]",
        "raw_llm_response": "",
        "prompt_preview": "",
        "error_message": "",
        **ref,
    }


def _apply_prediction(row: Dict[str, Any], pred: Dict[str, Any]) -> None:
    row["klasse"] = pred["klasse"] if pred["klasse"] is not None else ""
    row["label"] = pred.get("label", "")
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


def process_single_case(
    case: ClinicalCase,
    ref_lookup: ReferenceLookup,
    *,
    dry_run: bool = False,
    llm_call: Optional[Callable[[list], str]] = None,
) -> Dict[str, Any]:
    row = _base_row(case, ref_lookup)

    if dry_run:
        row["status"] = "dry_run"
        row["prompt_preview"] = prompt_preview(case)
        return row

    raw = ""
    try:
        if llm_call is None:
            from src.models.llm_interface import call_llm

            llm_call = call_llm
        raw = llm_call(build_messages(case)) or ""
        row["raw_llm_response"] = raw
    except Exception as exc:
        row["status"] = "llm_failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("LLM failed case_id=%s", case.case_id)
        return row

    pred, err = parse_hemorrhage_response(raw, context=f"hemorrhage_case:{case.case_id}")
    _apply_prediction(row, pred)

    if err:
        row["status"] = "parse_failed"
        row["error_message"] = err
    else:
        row["status"] = "success"

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

    if not cases:
        result.errors.append("no_cases_built")
        result.summary_lines = ["No cases to process.", *result.errors]
        return result

    ref_lookup, ref_errors = build_reference_lookup(reference_path)
    result.errors.extend(ref_errors)

    work = list(cases)
    if case_id:
        work = [c for c in work if c.case_id == case_id]
        if not work:
            result.errors.append(f"case_id_not_found: {case_id}")
            result.summary_lines = [f"Case {case_id!r} not found among {len(cases)} cases."]
            return result
    if limit is not None and limit > 0:
        work = work[:limit]

    total = len(work)
    rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(work, start=1):
        print(f"Processing case {idx}/{total} | case_id={case.case_id} | dry_run={dry_run}")
        row = process_single_case(case, ref_lookup, dry_run=dry_run, llm_call=llm_call)
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

    write_predictions_csv(rows, out_path)
    result.cases_processed = len(rows)

    result.summary_lines = [
        "Hemorrhage case pipeline (prototype)",
        f"reports_file={reports_file}",
        f"cases_total_loaded={len(cases)}",
        f"cases_processed={result.cases_processed}",
        f"dry_run={dry_run}",
        f"success={result.success_count}",
        f"parse_failed={result.parse_failed_count}",
        f"llm_failed={result.llm_failed_count}",
        f"dry_run_rows={result.dry_run_count}",
        f"output={out_path}",
        f"input_rows={stats.input_rows}",
        f"cases_incomplete={stats.cases_incomplete}",
    ]
    if result.errors:
        result.summary_lines.append("warnings/errors:")
        for e in result.errors:
            result.summary_lines.append(f"  - {e}")

    return result
