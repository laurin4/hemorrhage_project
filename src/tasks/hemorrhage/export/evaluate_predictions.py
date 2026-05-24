"""Backward-compatible re-export; implementation lives in evaluation.runner."""

from src.tasks.hemorrhage.evaluation.runner import (
    EvaluationResult,
    build_confusion_matrix_rows,
    build_error_cases_df,
    build_metrics_record,
    build_summary_lines,
    compute_binary_metrics,
    compute_counts,
    generate_plots,
    run_evaluate_predictions,
    safe_div,
)

__all__ = [
    "EvaluationResult",
    "build_confusion_matrix_rows",
    "build_error_cases_df",
    "build_metrics_record",
    "build_summary_lines",
    "compute_binary_metrics",
    "compute_counts",
    "generate_plots",
    "run_evaluate_predictions",
    "safe_div",
]
