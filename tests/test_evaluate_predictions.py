"""Tests for hemorrhage preliminary evaluation export."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.tasks.hemorrhage.evaluation.runner import (
    build_error_cases_df,
    build_metrics_record,
    build_subtype_by_reference_rows,
    compute_binary_metrics,
    compute_counts,
    compute_subtype_counts,
    run_evaluate_predictions,
    safe_div,
)


def _sample_review_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "c1",
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "reference_status": "hemorrhagic",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "prediction_vs_reference": "TP",
                "error_type": "correct_positive",
                "sicherheit": "hoch",
                "begruendung": "Akute Blutung.",
                "evidence_summary": "OP: Blutung",
            },
            {
                "case_id": "c2",
                "excel_pid": "2",
                "excel_opdat": "2024-01-02",
                "reference_status": "non_hemorrhagic",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "prediction_vs_reference": "TN",
                "error_type": "correct_negative",
                "sicherheit": "mittel",
                "begruendung": "Keine relevante Blutung.",
                "evidence_summary": "",
            },
            {
                "case_id": "c3",
                "excel_pid": "3",
                "excel_opdat": "2024-01-03",
                "reference_status": "hemorrhagic",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "prediction_vs_reference": "FN",
                "error_type": "false_negative",
                "sicherheit": "niedrig",
                "begruendung": "Fehlklassifikation.",
                "evidence_summary": "Eintritt: alt",
            },
            {
                "case_id": "c4",
                "excel_pid": "4",
                "excel_opdat": "2024-01-04",
                "reference_status": "non_hemorrhagic",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "prediction_vs_reference": "FP",
                "error_type": "false_positive",
                "sicherheit": "hoch",
                "begruendung": "Kavernom FP.",
                "evidence_summary": "OP: geblutetes Kavernom",
            },
            {
                "case_id": "c5",
                "excel_pid": "5",
                "excel_opdat": "2024-01-05",
                "reference_status": "verify_only",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "prediction_vs_reference": "reference_unknown",
                "error_type": "unknown_reference",
                "sicherheit": "mittel",
                "begruendung": "",
                "evidence_summary": "",
            },
            {
                "case_id": "c6",
                "excel_pid": "6",
                "excel_opdat": "2024-01-06",
                "reference_status": "unknown",
                "status": "success",
                "klasse": 0,
                "label": "nicht_hämorrhagisch",
                "prediction_vs_reference": "reference_unknown",
                "error_type": "unknown_reference",
                "sicherheit": "mittel",
                "begruendung": "",
                "evidence_summary": "",
            },
            {
                "case_id": "c7",
                "excel_pid": "7",
                "excel_opdat": "2024-01-07",
                "reference_status": "hemorrhagic",
                "status": "parse_failed",
                "klasse": "",
                "label": "",
                "prediction_vs_reference": "prediction_missing",
                "error_type": "pipeline_failure",
                "sicherheit": "",
                "begruendung": "",
                "evidence_summary": "",
            },
            {
                "case_id": "c8",
                "excel_pid": "8",
                "excel_opdat": "2024-01-08",
                "reference_status": "non_hemorrhagic",
                "status": "llm_failed",
                "klasse": "",
                "label": "",
                "prediction_vs_reference": "prediction_missing",
                "error_type": "pipeline_failure",
                "sicherheit": "",
                "begruendung": "",
                "evidence_summary": "",
            },
        ]
    )


def test_tp_tn_fp_fn_metric_calculation():
    df = _sample_review_rows()
    counts = compute_counts(df, include_verify_as_negative=False)
    assert counts["TP"] == 1
    assert counts["TN"] == 1
    assert counts["FP"] == 1
    assert counts["FN"] == 1
    assert counts["evaluated_cases"] == 4
    assert counts["labeled_cases"] == 6

    metrics = compute_binary_metrics(counts)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["NPV"] == pytest.approx(0.5)
    assert metrics["F1"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_verify_only_excluded_by_default():
    df = _sample_review_rows()
    counts = compute_counts(df, include_verify_as_negative=False)
    assert counts["excluded_verify_only"] == 1
    assert counts["evaluated_cases"] == 4


def test_zero_division_safe():
    assert safe_div(1, 0) is None
    metrics = compute_binary_metrics({"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    assert metrics["accuracy"] is None
    assert metrics["F1"] is None
    assert metrics["balanced_accuracy"] is None


def test_sensitivity_mode_verify_as_negative():
    df = _sample_review_rows()
    counts = compute_counts(df, include_verify_as_negative=True)
    assert counts["evaluated_cases"] == 5
    assert counts["TN"] == 2
    assert counts["excluded_verify_only"] == 0


def test_output_files_created(tmp_path: Path):
    review_path = tmp_path / "review.csv"
    confusion_path = tmp_path / "confusion.csv"
    out_dir = tmp_path / "evaluation"

    df = _sample_review_rows()
    df.to_csv(review_path, index=False)
    df[["case_id", "error_type", "prediction_vs_reference"]].to_csv(
        confusion_path, index=False
    )

    def _fake_plots(review_df, confusion_df, counts, plots_dir, *, include_verify_as_negative=False):
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "confusion_matrix.png").write_text("png", encoding="utf-8")
        (plots_dir / "reference_status_distribution.png").write_text("png", encoding="utf-8")
        return [], []

    with patch(
        "src.tasks.hemorrhage.evaluation.runner.generate_plots",
        side_effect=_fake_plots,
    ):
        result = run_evaluate_predictions(
            review_path=review_path,
            confusion_path=confusion_path,
            output_dir=out_dir,
            include_verify_as_negative=True,
        )
    assert not result.errors

    assert (out_dir / "hemorrhage_metrics_summary.csv").exists()
    assert (out_dir / "hemorrhage_metrics_summary.txt").exists()
    assert (out_dir / "hemorrhage_metrics_summary.md").exists()
    assert (out_dir / "hemorrhage_confusion_matrix.csv").exists()
    assert (out_dir / "hemorrhage_error_cases.csv").exists()
    assert (out_dir / "plots" / "confusion_matrix.png").exists()
    assert (out_dir / "plots" / "reference_status_distribution.png").exists()
    assert (out_dir / "hemorrhage_metrics_summary_verify_as_negative.csv").exists()

    summary = (out_dir / "hemorrhage_metrics_summary.txt").read_text(encoding="utf-8")
    assert "Hemorrhage Preliminary Evaluation" in summary
    assert "Dataset overview" in summary
    assert "Performance metrics" in summary
    assert "Accuracy:" in summary and "%" in summary
    assert "Interpretation" in summary
    assert "FN detailed review" in summary
    md = (out_dir / "hemorrhage_metrics_summary.md").read_text(encoding="utf-8")
    assert "# Hemorrhage Preliminary Evaluation" in md
    assert "| TP |" in md or "(TP)" in md

    errors = pd.read_csv(out_dir / "hemorrhage_error_cases.csv")
    assert set(errors["prediction_vs_reference"]) <= {"FP", "FN", "prediction_missing"}
    assert len(errors) == 4


def test_error_cases_include_fp_fn_and_missing():
    df = _sample_review_rows()
    errors = build_error_cases_df(df)
    assert len(errors) == 4
    assert set(errors["case_id"]) == {"c3", "c4", "c7", "c8"}


def _subtype_review_rows() -> pd.DataFrame:
    df = _sample_review_rows()
    df["predicted_haemorrhage_subtype"] = ""
    # c1 (TP, hämorrhagisch) → akut, c4 (FP, hämorrhagisch) → nicht_akut
    df.loc[df["case_id"] == "c1", "predicted_haemorrhage_subtype"] = "akut"
    df.loc[df["case_id"] == "c4", "predicted_haemorrhage_subtype"] = "nicht_akut"
    return df


def test_subtype_counts_generated():
    df = _subtype_review_rows()
    counts = compute_subtype_counts(df)
    assert counts["akut"] == 1
    assert counts["nicht_akut"] == 1
    assert counts["historisch"] == 0
    # only hämorrhagisch rows are counted (c1, c4); none missing here
    assert counts["unbekannt"] == 0


def test_subtype_missing_counts_as_unbekannt():
    df = _sample_review_rows()  # hämorrhagisch rows have no subtype column
    counts = compute_subtype_counts(df)
    # c1 (TP) and c4 (FP) are hämorrhagisch with no subtype → unbekannt
    assert counts["unbekannt"] == 2
    assert counts["akut"] == 0


def test_historical_subtype_counts_as_binary_positive():
    """klasse=1 + subtype historisch against a hemorrhagic reference is a TP."""
    df = pd.DataFrame(
        [
            {
                "case_id": "h1",
                "excel_pid": "1",
                "excel_opdat": "d1",
                "reference_status": "hemorrhagic",
                "status": "success",
                "klasse": 1,
                "label": "hämorrhagisch",
                "predicted_haemorrhage_subtype": "historisch",
                "prediction_vs_reference": "TP",
                "error_type": "correct_positive",
                "sicherheit": "mittel",
                "begruendung": "Status nach Blutung.",
                "evidence_summary": "",
            }
        ]
    )
    counts = compute_counts(df, include_verify_as_negative=False)
    assert counts["TP"] == 1
    assert counts["FN"] == 0
    subtype = compute_subtype_counts(df)
    assert subtype["historisch"] == 1


def test_subtype_does_not_affect_binary_metrics():
    base = compute_counts(_sample_review_rows(), include_verify_as_negative=False)
    with_subtype = compute_counts(_subtype_review_rows(), include_verify_as_negative=False)
    for key in ("TP", "TN", "FP", "FN", "evaluated_cases", "labeled_cases"):
        assert base[key] == with_subtype[key]


def test_subtype_by_reference_only_hemorrhagic_predictions():
    rows = build_subtype_by_reference_rows(_subtype_review_rows())
    # both hämorrhagisch predictions: c1 ref hemorrhagic/akut, c4 ref non_hemorrhagic/nicht_akut
    keys = {(r["reference_status"], r["haemorrhage_subtype"]) for r in rows}
    assert ("hemorrhagic", "akut") in keys
    assert ("non_hemorrhagic", "nicht_akut") in keys


def test_subtype_outputs_written(tmp_path: Path):
    review_path = tmp_path / "review.csv"
    confusion_path = tmp_path / "confusion.csv"
    out_dir = tmp_path / "evaluation"

    df = _subtype_review_rows()
    df.to_csv(review_path, index=False)
    df[["case_id", "error_type", "prediction_vs_reference"]].to_csv(confusion_path, index=False)

    def _fake_plots(review_df, confusion_df, counts, plots_dir, *, include_verify_as_negative=False):
        plots_dir.mkdir(parents=True, exist_ok=True)
        return [], []

    with patch(
        "src.tasks.hemorrhage.evaluation.runner.generate_plots",
        side_effect=_fake_plots,
    ):
        result = run_evaluate_predictions(
            review_path=review_path,
            confusion_path=confusion_path,
            output_dir=out_dir,
        )
    assert not result.errors
    assert (out_dir / "hemorrhage_subtype_distribution.csv").exists()
    assert (out_dir / "hemorrhage_subtype_by_reference_status.csv").exists()
    summary = (out_dir / "hemorrhage_metrics_summary.txt").read_text(encoding="utf-8")
    assert "Hemorrhage subtype analysis" in summary
    assert "akut" in summary
    md = (out_dir / "hemorrhage_metrics_summary.md").read_text(encoding="utf-8")
    assert "Hemorrhage subtype analysis" in md


def test_build_metrics_record_rounding():
    df = _sample_review_rows()
    record = build_metrics_record(df)
    assert record["accuracy"] == 0.5
