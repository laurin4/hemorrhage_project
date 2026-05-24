"""Tests for hemorrhage prediction review export."""

import json
from pathlib import Path

import pandas as pd

from src.tasks.hemorrhage.export.prediction_review import (
    CONFUSION_CSV_COLUMNS,
    DETAILED_ERROR_REVIEW_COLUMNS,
    REVIEW_CSV_COLUMNS,
    compute_prediction_vs_reference,
    derive_error_type,
    derive_reference_status,
    flatten_evidence_summary,
    run_build_prediction_review,
)


def test_derive_reference_status():
    assert derive_reference_status(1, 0, 0) == "hemorrhagic"
    assert derive_reference_status(0, 1, 0) == "non_hemorrhagic"
    assert derive_reference_status(0, 0, 1) == "verify_only"
    assert derive_reference_status(1, 1, 0) == "inconsistent"
    assert derive_reference_status(0, 0, 0) == "unknown"


def test_prediction_vs_reference_verify_only_not_tn():
    assert compute_prediction_vs_reference("verify_only", "success", 0, "nicht_hämorrhagisch") == "reference_unknown"
    assert compute_prediction_vs_reference("non_hemorrhagic", "success", 0, "nicht_hämorrhagisch") == "TN"
    assert compute_prediction_vs_reference("hemorrhagic", "success", 1, "hämorrhagisch") == "TP"
    assert compute_prediction_vs_reference("hemorrhagic", "success", 0, "nicht_hämorrhagisch") == "FN"
    assert compute_prediction_vs_reference("non_hemorrhagic", "success", 1, "hämorrhagisch") == "FP"


def test_flatten_evidence_markdown_style():
    evidenz = [
        {
            "berichttyp": "03 Austrittsbericht",
            "feld": "diag",
            "textstelle": "frische Blutung",
            "interpretation": "aktuell relevant",
        }
    ]
    summary = flatten_evidence_summary(json.dumps(evidenz, ensure_ascii=False))
    assert "Austrittsbericht/diag" in summary
    assert "frische Blutung" in summary


def test_build_prediction_review_export(tmp_path: Path):
    preds = pd.DataFrame(
        [
            {
                "case_id": "case_1__2024-01-01__F1",
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "Aktuelle Blutung.",
                "evidenz_json": json.dumps(
                    [{"berichttyp": "01 Operationsbericht", "feld": "diag", "textstelle": "Blutung"}]
                ),
                "historische_blutung_erwaehnt": False,
                "historische_blutung_als_aktuell_gewertet": False,
                "reference_haemorrhagisch": "1",
                "reference_nicht_haemorrhagisch": "0",
                "reference_verify_vaskulaer": "",
            },
            {
                "case_id": "case_2__2024-01-02__F2",
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "F2",
                "status": "dry_run",
                "klasse": "",
                "label": "",
                "sicherheit": "",
                "begruendung": "",
                "evidenz_json": "[]",
                "historische_blutung_erwaehnt": "",
                "historische_blutung_als_aktuell_gewertet": "",
                "reference_haemorrhagisch": "",
                "reference_nicht_haemorrhagisch": "",
                "reference_verify_vaskulaer": "1",
            },
        ]
    )
    pred_path = tmp_path / "preds.csv"
    out_path = tmp_path / "review.csv"
    conf_path = tmp_path / "confusion.csv"
    sum_path = tmp_path / "summary.txt"
    preds.to_csv(pred_path, index=False)

    result = run_build_prediction_review(
        predictions_path=pred_path,
        review_path=out_path,
        confusion_path=conf_path,
        summary_path=sum_path,
    )
    assert result.rows_written == 2
    assert out_path.exists()
    assert conf_path.exists()
    assert sum_path.exists()

    review = pd.read_csv(out_path)
    assert list(review.columns) == REVIEW_CSV_COLUMNS
    confusion = pd.read_csv(conf_path)
    assert list(confusion.columns) == CONFUSION_CSV_COLUMNS
    assert "begruendung" not in confusion.columns
    assert "evidence_summary" not in confusion.columns
    by_id = {r["case_id"]: r for _, r in review.iterrows()}
    assert by_id["case_1__2024-01-01__F1"]["prediction_vs_reference"] == "TP"
    assert by_id["case_2__2024-01-02__F2"]["reference_status"] == "verify_only"
    assert by_id["case_2__2024-01-02__F2"]["prediction_vs_reference"] == "reference_unknown"

    summary = sum_path.read_text(encoding="utf-8")
    assert "NOT final evaluation" in summary
    assert "verify_only=1" in summary
    assert "prediction_missing=" in summary
    assert "reference_unknown=" in summary
    assert "false_negative_review_path=" in summary
    assert "false_positive_review_path=" in summary


def test_fn_fp_detailed_exports(tmp_path: Path):
    preds = pd.DataFrame(
        [
            {
                "case_id": "fn_case",
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "sicherheit": "niedrig",
                "begruendung": "Modell sah keine akute Blutung.",
                "evidenz_json": '[{"berichttyp":"01 Operationsbericht","feld":"diag","textstelle":"alt"}]',
                "unsicherheitsgruende_json": '["historische Blutung"]',
                "historische_blutung_erwaehnt": True,
                "historische_blutung_als_aktuell_gewertet": False,
                "raw_llm_response": '{"klasse":0}',
                "reference_haemorrhagisch": "1",
                "reference_nicht_haemorrhagisch": "0",
                "reference_verify_vaskulaer": "",
            },
            {
                "case_id": "fp_case",
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "F2",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "Geblutetes Kavernom fälschlich als akut.",
                "evidenz_json": '[{"berichttyp":"01 Operationsbericht","feld":"diag","textstelle":"geblutetes Kavernom"}]',
                "unsicherheitsgruende_json": "[]",
                "historische_blutung_erwaehnt": True,
                "historische_blutung_als_aktuell_gewertet": True,
                "raw_llm_response": '{"klasse":1}',
                "reference_haemorrhagisch": "0",
                "reference_nicht_haemorrhagisch": "1",
                "reference_verify_vaskulaer": "",
            },
            {
                "case_id": "tn_case",
                "excel_pid": "3",
                "excel_opdat": "2024-01-03",
                "opber_fallnr": "F3",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "Korrekt negativ.",
                "evidenz_json": "[]",
                "unsicherheitsgruende_json": "[]",
                "reference_haemorrhagisch": "0",
                "reference_nicht_haemorrhagisch": "1",
                "reference_verify_vaskulaer": "",
            },
        ]
    )
    pred_path = tmp_path / "preds.csv"
    fn_path = tmp_path / "fn.csv"
    fp_path = tmp_path / "fp.csv"
    preds.to_csv(pred_path, index=False)

    result = run_build_prediction_review(
        predictions_path=pred_path,
        review_path=tmp_path / "review.csv",
        confusion_path=tmp_path / "confusion.csv",
        summary_path=tmp_path / "summary.txt",
        false_negative_path=fn_path,
        false_positive_path=fp_path,
    )
    assert result.fn_count == 1
    assert result.fp_count == 1
    assert fn_path.exists()
    assert fp_path.exists()

    fn_df = pd.read_csv(fn_path)
    fp_df = pd.read_csv(fp_path)
    assert list(fn_df.columns) == DETAILED_ERROR_REVIEW_COLUMNS
    assert list(fp_df.columns) == DETAILED_ERROR_REVIEW_COLUMNS
    assert len(fn_df) == 1
    assert len(fp_df) == 1
    assert fn_df.iloc[0]["prediction_vs_reference"] == "FN"
    assert fp_df.iloc[0]["prediction_vs_reference"] == "FP"
    assert fn_df.iloc[0]["error_type"] == "false_negative"
    assert fp_df.iloc[0]["error_type"] == "false_positive"
    assert "evidenz_json" in fn_df.columns
    assert fn_df.iloc[0]["raw_llm_response"] == '{"klasse":0}'
    assert "geblutetes Kavernom" in fp_df.iloc[0]["evidenz_json"]


def test_error_type_mapping():
    assert derive_error_type("TP") == "correct_positive"
    assert derive_error_type("FN") == "false_negative"
    assert derive_error_type("reference_unknown") == "unknown_reference"
    assert derive_error_type("prediction_missing") == "pipeline_failure"


def test_confusion_sort_fn_before_tp(tmp_path: Path):
    preds = pd.DataFrame(
        [
            {
                "case_id": "tp_case",
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "",
                "evidenz_json": "[]",
                "reference_haemorrhagisch": "1",
                "reference_nicht_haemorrhagisch": "0",
                "reference_verify_vaskulaer": "",
            },
            {
                "case_id": "fn_case",
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "F2",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "",
                "evidenz_json": "[]",
                "reference_haemorrhagisch": "1",
                "reference_nicht_haemorrhagisch": "0",
                "reference_verify_vaskulaer": "",
            },
        ]
    )
    pred_path = tmp_path / "preds.csv"
    conf_path = tmp_path / "confusion.csv"
    preds.to_csv(pred_path, index=False)

    run_build_prediction_review(
        predictions_path=pred_path,
        review_path=tmp_path / "review.csv",
        confusion_path=conf_path,
        summary_path=tmp_path / "summary.txt",
    )
    confusion = pd.read_csv(conf_path)
    assert confusion.iloc[0]["prediction_vs_reference"] == "FN"
    assert confusion.iloc[0]["error_type"] == "false_negative"
    assert confusion.iloc[1]["prediction_vs_reference"] == "TP"


def test_only_mismatches_filter(tmp_path: Path):
    preds = pd.DataFrame(
        [
            {
                "case_id": "a",
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "",
                "evidenz_json": "[]",
                "reference_haemorrhagisch": "0",
                "reference_nicht_haemorrhagisch": "1",
                "reference_verify_vaskulaer": "",
            },
            {
                "case_id": "b",
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "F2",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "sicherheit": "hoch",
                "begruendung": "",
                "evidenz_json": "[]",
                "reference_haemorrhagisch": "0",
                "reference_nicht_haemorrhagisch": "1",
                "reference_verify_vaskulaer": "",
            },
        ]
    )
    pred_path = tmp_path / "preds.csv"
    preds.to_csv(pred_path, index=False)

    result = run_build_prediction_review(
        predictions_path=pred_path,
        review_path=tmp_path / "review.csv",
        confusion_path=tmp_path / "confusion.csv",
        summary_path=tmp_path / "summary.txt",
        only_mismatches=True,
    )
    assert result.rows_written == 1
    review = pd.read_csv(tmp_path / "review.csv")
    assert review.iloc[0]["prediction_vs_reference"] == "FP"
