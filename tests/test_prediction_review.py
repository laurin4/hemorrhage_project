"""Tests for hemorrhage prediction review export."""

import json
from pathlib import Path

import pandas as pd

from src.tasks.hemorrhage.export.prediction_review import (
    REVIEW_CSV_COLUMNS,
    build_review_row,
    compute_prediction_vs_reference,
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
    sum_path = tmp_path / "summary.txt"
    preds.to_csv(pred_path, index=False)

    result = run_build_prediction_review(
        predictions_path=pred_path,
        review_path=out_path,
        summary_path=sum_path,
    )
    assert result.rows_written == 2
    assert out_path.exists()
    assert sum_path.exists()

    review = pd.read_csv(out_path)
    assert list(review.columns) == REVIEW_CSV_COLUMNS
    by_id = {r["case_id"]: r for _, r in review.iterrows()}
    assert by_id["case_1__2024-01-01__F1"]["prediction_vs_reference"] == "TP"
    assert by_id["case_2__2024-01-02__F2"]["reference_status"] == "verify_only"
    assert by_id["case_2__2024-01-02__F2"]["prediction_vs_reference"] == "reference_unknown"

    summary = sum_path.read_text(encoding="utf-8")
    assert "NOT final evaluation" in summary
    assert "verify_only=1" in summary


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
        summary_path=tmp_path / "summary.txt",
        only_mismatches=True,
    )
    assert result.rows_written == 1
    review = pd.read_csv(tmp_path / "review.csv")
    assert review.iloc[0]["prediction_vs_reference"] == "FP"
