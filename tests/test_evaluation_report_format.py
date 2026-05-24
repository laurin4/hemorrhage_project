"""Tests for human-readable evaluation report formatting."""

from pathlib import Path

from src.tasks.hemorrhage.evaluation.report_format import (
    EvaluationReportPaths,
    build_ascii_confusion_matrix,
    build_readable_reports,
    format_pct,
)


def test_format_pct_one_decimal():
    assert format_pct(0.355) == "35.5%"
    assert format_pct(None) == "n/a"


def test_ascii_confusion_matrix():
    lines = build_ascii_confusion_matrix(tp=5, tn=6, fp=2, fn=18)
    text = "\n".join(lines)
    assert "TN" in text and "18" in text


def test_readable_reports_contain_sections(tmp_path: Path):
    metrics = {
        "total_cases": 114,
        "labeled_cases": 32,
        "evaluated_cases": 31,
        "excluded_verify_only": 82,
        "excluded_unknown": 0,
        "excluded_inconsistent": 0,
        "excluded_prediction_missing": 1,
        "parse_failed": 0,
        "llm_failed": 0,
        "TP": 5,
        "TN": 6,
        "FP": 2,
        "FN": 18,
        "accuracy": 0.354839,
        "sensitivity": 0.217391,
        "specificity": 0.75,
        "precision": 0.714286,
        "NPV": 0.25,
        "F1": 0.333333,
        "balanced_accuracy": 0.483696,
    }
    paths = EvaluationReportPaths(
        review_csv=tmp_path / "review.csv",
        confusion_review_csv=tmp_path / "confusion.csv",
        false_negative_review_csv=tmp_path / "fn.csv",
        false_positive_review_csv=tmp_path / "fp.csv",
        metrics_csv=tmp_path / "metrics.csv",
        confusion_matrix_csv=tmp_path / "cm.csv",
        error_cases_csv=tmp_path / "errors.csv",
        plots_dir=tmp_path / "plots",
    )
    txt, md = build_readable_reports(metrics, paths)
    assert "FN: 18" in txt
    assert "Sensitivity (Recall):  21.7%" in txt
    assert "Class imbalance" in txt or "Reference hemorrhagic" in txt
    assert "## Confusion matrix" in md
    assert "geblutetes Kavernom" not in md  # interpretation is heuristic, not hardcoded
