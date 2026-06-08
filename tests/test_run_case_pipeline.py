"""Tests for hemorrhage case-level inference pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.core.case.keys import CaseKey
from src.core.case.models import CaseReport, build_clinical_case
from src.tasks.hemorrhage.constants import (
    EXPECTED_REPORT_TYPUS_CODES,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.inference.parse import parse_hemorrhage_response
from src.tasks.hemorrhage.inference.prompt import build_user_prompt, prompt_preview
from src.tasks.hemorrhage.inference.runner import (
    PREDICTION_CSV_COLUMNS,
    process_single_case,
    run_hemorrhage_case_pipeline,
    write_predictions_csv,
)


def _incomplete_case():
    key = CaseKey("P1", "2024-01-01", "F1")
    reports = {
        TYPUS_OPERATIONSBERICHT: CaseReport(
            typus_code=TYPUS_OPERATIONSBERICHT,
            typus_label="01 Operationsbericht",
            report_text="[Diagnosen]\nCCM mit Blutung",
        )
    }
    return build_clinical_case(
        key,
        reports,
        expected_typus_codes=EXPECTED_REPORT_TYPUS_CODES,
    )


def test_prompt_incomplete_case():
    case = _incomplete_case()
    user = build_user_prompt(case)
    assert "01 Operationsbericht" in user
    assert "CCM" in user
    preview = prompt_preview(case, max_chars=500)
    assert len(preview) <= 501


def test_parse_success():
    raw = json.dumps(
        {
            "klasse": 1,
            "label": "hämorrhagisch",
            "sicherheit": "mittel",
            "begruendung": "Aktuelle Blutung im OP-Bericht.",
            "evidenz": [],
            "historische_blutung_erwaehnt": False,
            "historische_blutung_als_aktuell_gewertet": False,
            "unsicherheitsgruende": [],
        },
        ensure_ascii=False,
    )
    result = parse_hemorrhage_response(raw, context="test")
    assert result.success
    assert result.prediction["klasse"] == 1


def test_parse_failure():
    result = parse_hemorrhage_response("not json at all", context="test")
    assert not result.success
    assert result.prediction["klasse"] is None


def test_parse_json_inside_markdown_fence():
    payload = {
        "klasse": 0,
        "label": "nicht_hämorrhagisch",
        "sicherheit": "hoch",
        "begruendung": "Keine aktuelle Blutung.",
        "evidenz": [],
        "historische_blutung_erwaehnt": False,
        "historische_blutung_als_aktuell_gewertet": False,
        "unsicherheitsgruende": [],
    }
    raw = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    result = parse_hemorrhage_response(raw, context="test")
    assert result.success
    assert result.prediction["klasse"] == 0


def test_dry_run_does_not_invoke_parser(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "diag": "Test",
            }
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"

    with patch(
        "src.tasks.hemorrhage.inference.runner.parse_binary_response",
        side_effect=AssertionError("parser must not run in dry-run"),
    ), patch(
        "src.tasks.hemorrhage.inference.runner.parse_subtype_response",
        side_effect=AssertionError("parser must not run in dry-run"),
    ):
        result = run_hemorrhage_case_pipeline(
            reports_path=xlsx,
            output_path=out,
            dry_run=True,
        )
    assert result.dry_run_count == 1


def test_hemorrhage_inference_imports_resolve():
    import src.tasks.hemorrhage.inference.llm_client  # noqa: F401
    import src.tasks.hemorrhage.inference.parse  # noqa: F401
    import src.tasks.hemorrhage.inference.runner  # noqa: F401
    import src.tasks.hemorrhage.run_case_pipeline  # noqa: F401


def test_dry_run_pipeline(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "diag": "Test",
            }
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"

    result = run_hemorrhage_case_pipeline(
        reports_path=xlsx,
        output_path=out,
        dry_run=True,
    )
    assert result.cases_processed == 1
    assert result.dry_run_count == 1
    assert out.exists()
    saved = pd.read_csv(out)
    assert saved.iloc[0]["status"] == "dry_run"
    assert saved.iloc[0]["prompt_preview"]


def test_pipeline_continues_after_llm_failure(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "diag": "A",
            },
            {
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "F2",
                "typus": "02 Eintrittsbericht",
                "diag": "B",
            },
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"

    call_count = {"n": 0}

    def flaky_llm(_messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated LLM failure")
        return json.dumps(
            {
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "ok",
                "evidenz": [],
                "historische_blutung_erwaehnt": False,
                "historische_blutung_als_aktuell_gewertet": False,
                "unsicherheitsgruende": [],
            }
        )

    result = run_hemorrhage_case_pipeline(
        reports_path=xlsx,
        output_path=out,
        llm_call=flaky_llm,
    )
    assert result.cases_processed == 2
    assert result.llm_failed_count == 1
    assert result.success_count == 1
    saved = pd.read_csv(out)
    assert len(saved) == 2


def test_output_schema_columns(tmp_path: Path):
    row = process_single_case(_incomplete_case(), {}, dry_run=True)
    write_predictions_csv([row], tmp_path / "out.csv")
    df = pd.read_csv(tmp_path / "out.csv")
    assert list(df.columns) == PREDICTION_CSV_COLUMNS
