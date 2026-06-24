import os
from pathlib import Path

# Projektbasis
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Active task: "hemorrhage" (case-centric) | "delirium" (legacy report-centric pipeline)
PROJECT_TASK = os.environ.get("PROJECT_TASK", "hemorrhage").strip().lower()

# Daten
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_DIR
REAL_RAW_DIR = REAL_DATA_DIR / "raw"
ANONYMIZED_DIR = DATA_DIR / "anonymized"
DIAGNOSIS_EXAMPLES_DIR = ANONYMIZED_DIR / "beispiele"
STRUCTURED_DIR = DATA_DIR / "structured"
STRUCTURED_RAW_DIR = STRUCTURED_DIR / "raw"

# Default production inputs (CSV unter data/raw fuer Ubuntu/local parity).
# Set DATA_MODE = "synthetic" only for offline regression tests (CSV generator outputs).
DATA_MODE = "real"  # allowed: "real", "synthetic"


# Default: no cap — process the full evaluatable Berichte corpus (thesis / validation runs).
# Set MAX_REPORTS=<n> in the environment for pilot or dev slices only.
DEFAULT_MAX_REPORTS: int | None = None


def parse_max_reports_env(raw: str | None = None) -> int | None:
    """
    Parse MAX_REPORTS from *raw* or from the environment.

    - Unset / blank → ``DEFAULT_MAX_REPORTS`` (``None`` = full corpus).
    - ``all`` (case-insensitive) → ``None`` (no limit).
    - Otherwise a positive integer cap.

    Raises ValueError for invalid strings or non-positive integers (except ``all``).
    """
    if raw is None:
        raw = os.environ.get("MAX_REPORTS", "")
    raw = raw.strip()
    if not raw:
        return DEFAULT_MAX_REPORTS
    if raw.lower() == "all":
        return None
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (full corpus)."
        ) from exc
    if n <= 0:
        raise ValueError(
            "MAX_REPORTS must be 'all', a positive integer, or unset (full corpus)."
        )
    return n


def _max_reports_from_environment() -> int | None:
    """``MAX_REPORTS`` env (see ``parse_max_reports_env``); default is full corpus."""
    return parse_max_reports_env()


MAX_REPORTS = _max_reports_from_environment()

# TEMPORARY PRESENTATION MODE — baseline_composite definition (revert for thesis default):
#   "OR"  = (ICDSC>=4) OR ICD10  — broader / sensitive thesis baseline
#   "AND" = (ICDSC>=4) AND ICD10 — stricter "secure delir cases" for presentation/demo
BASELINE_COMPOSITE_MODE = "AND"  # allowed: "OR", "AND"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
VALIDATION_DIR = OUTPUTS_DIR / "validation"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
ANALYSIS_TABLES_DIR = ANALYSIS_DIR / "tables"
ANALYSIS_PLOTS_DIR = ANALYSIS_DIR / "plots"
ANALYSIS_REPORTS_DIR = ANALYSIS_DIR / "reports"
EXPLORATION_DIR = ANALYSIS_DIR / "exploration"
EXPLORATION_TABLES_DIR = EXPLORATION_DIR / "tables"
EXPLORATION_PLOTS_DIR = EXPLORATION_DIR / "plots"
EXPLORATION_REPORTS_DIR = EXPLORATION_DIR / "reports"
ANALYSIS_EVALUATION_DIR = ANALYSIS_DIR / "evaluation"
ANALYSIS_EVALUATION_TABLES_DIR = ANALYSIS_EVALUATION_DIR / "tables"
ANALYSIS_EVALUATION_PLOTS_DIR = ANALYSIS_EVALUATION_DIR / "plots"
PREPARED_DATA_DIR = OUTPUTS_DIR / "prepared"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Final production raw inputs (DATA_MODE=real): Berichte.csv, ICD.csv, ICDSC.csv only.
# Legacy Diagnosenliste.csv is not used in the active pipeline (see LEGACY_DIAGNOSIS_INPUT_PATH).

# Input paths per mode (single source of truth; no duplicated path logic)
_MODE_INPUTS = {
    "real": {
        "icd10": REAL_RAW_DIR / "ICD.csv",
        "icdsc": REAL_RAW_DIR / "ICDSC.csv",
        "berichte_csv": REAL_RAW_DIR / "Berichte.csv",
    },
    "synthetic": {
        "icd10": STRUCTURED_RAW_DIR / "synthetic_icd10.csv",
        "icdsc": STRUCTURED_RAW_DIR / "synthetic_icdsc.csv",
        "berichte_csv": STRUCTURED_RAW_DIR / "synthetic_berichte.csv",
        # Legacy text source for offline regression only (INPUT_MODE=diagnosis).
        "diagnosis": DIAGNOSIS_EXAMPLES_DIR / "synthetic_diagnoses.csv",
    },
}

# Legacy path (not required for production). Former Diagnosenliste.csv location.
LEGACY_DIAGNOSIS_INPUT_PATH = REAL_RAW_DIR / "Diagnosenliste.csv"

if DATA_MODE not in _MODE_INPUTS:
    raise ValueError(
        f"Invalid DATA_MODE='{DATA_MODE}'. Allowed values: {sorted(_MODE_INPUTS)}"
    )

_paths = _MODE_INPUTS[DATA_MODE]
ICD10_PATH = _paths["icd10"]
ICDSC_PATH = _paths["icdsc"]
BERICHTE_INPUT_PATH = _paths["berichte_csv"]
# Legacy diagnosis list — only defined in synthetic mode; use Berichte.csv in production.
DIAGNOSIS_INPUT_PATH = _paths.get("diagnosis")
REPORT_ID_MAPPING_PATH = STRUCTURED_DIR / "report_patient_ids.csv"

STRUCTURED_BASELINE_PATH = BASELINE_DIR / "structured_baseline.csv"
REPORT_VS_BASELINE_PATH = COMPARISONS_DIR / "report_vs_baseline_comparison.csv"
REPORT_VS_BASELINE_EXCLUDED_PATH = (
    COMPARISONS_DIR / "report_vs_baseline_excluded_missing_baseline.csv"
)
EVALUATION_SUMMARY_PATH = EVALUATION_DIR / "evaluation_summary.csv"
EVALUATION_MULTICLASS_SUMMARY_PATH = EVALUATION_DIR / "evaluation_multiclass_summary.csv"
EVALUATION_CONFUSION_3CLASS_PATH = EVALUATION_DIR / "confusion_matrix_3class.csv"
EVALUATION_BINARY_BASELINES_DIR = EVALUATION_DIR / "binary_baselines"
EVALUATION_BINARY_BASELINES_TABLES_DIR = EVALUATION_BINARY_BASELINES_DIR / "tables"
EVALUATION_BINARY_BASELINES_PLOTS_DIR = EVALUATION_BINARY_BASELINES_DIR / "plots"
EVALUATION_BINARY_BASELINE_SUMMARY_PATH = (
    EVALUATION_BINARY_BASELINES_TABLES_DIR / "binary_baseline_summary.csv"
)
EVALUATION_BINARY_BASELINE_CONFUSION_COUNTS_PATH = (
    EVALUATION_BINARY_BASELINES_TABLES_DIR / "binary_baseline_confusion_counts.csv"
)
EVALUATION_BINARY_BASELINE_REPORT_PATH = (
    EVALUATION_BINARY_BASELINES_DIR / "report.txt"
)
PATIENT_LEVEL_REPORTS_PATH = PREPARED_DATA_DIR / "patient_level_reports.csv"
VALIDATION_RESULTS_CSV_PATH = VALIDATION_DIR / "validation_results.csv"
VALIDATION_SUMMARY_TXT_PATH = VALIDATION_DIR / "validation_summary.txt"


LLM_DEBUG_DIR = OUTPUTS_DIR / "logs" / "llm_debug"
SQLITE_PREDICTIONS_DB_PATH = LOGS_DIR / "prediction_run.sqlite"

# Cohort-limited inference (VALIDATION_COHORT_ONLY=true); does not overwrite full-run CSV.
VALIDATION_COHORT_PREDICTIONS_PATH = PREDICTIONS_DIR / "validation_cohort_predictions.csv"

# Field-level keyword analysis (Berichte.csv vs structured baselines)
FIELD_DELIRIUM_ANALYSIS_DIR = ANALYSIS_DIR / "field_delirium"
FIELD_DELIRIUM_TABLES_DIR = FIELD_DELIRIUM_ANALYSIS_DIR / "tables"
FIELD_DELIRIUM_PLOTS_DIR = FIELD_DELIRIUM_ANALYSIS_DIR / "plots"

# Pre-model data coverage (Berichte vs structured baseline)
DATA_COVERAGE_ANALYSIS_DIR = ANALYSIS_DIR / "data_coverage"
DATA_COVERAGE_TABLES_DIR = DATA_COVERAGE_ANALYSIS_DIR / "tables"
DATA_COVERAGE_PLOTS_DIR = DATA_COVERAGE_ANALYSIS_DIR / "plots"

# Error review (legacy dir; manual review export uses MANUAL_REVIEW_DIR)
ERROR_REVIEW_DIR = ANALYSIS_DIR / "error_review"
ERROR_REVIEW_TABLES_DIR = ERROR_REVIEW_DIR / "tables"
ERROR_REVIEW_PLOTS_DIR = ERROR_REVIEW_DIR / "plots"

# Manual scientific review (TP/TN/FP/FN samples per primary baseline)
MANUAL_REVIEW_DIR = ANALYSIS_DIR / "manual_review"

# Patient-level validation aggregation (report-level predictions → patient matrix)
PATIENT_LEVEL_ANALYSIS_DIR = ANALYSIS_DIR / "patient_level"
PATIENT_REPORTTYPE_MATRIX_PATH = PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix.csv"
PATIENT_REPORTTYPE_MATRIX_PREVIEW_PNG = (
    PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix_preview.png"
)
PATIENT_REPORTTYPE_MATRIX_PREVIEW_PDF = (
    PATIENT_LEVEL_ANALYSIS_DIR / "patient_reporttype_matrix_preview.pdf"
)

# Mixed manual validation sample (~100 patients)
MANUAL_VALIDATION_DIR = ANALYSIS_DIR / "manual_validation"
MANUAL_VALIDATION_SAMPLE_PATH = MANUAL_VALIDATION_DIR / "manual_validation_sample.csv"
MANUAL_ANNOTATION_SHEET_PATH = MANUAL_VALIDATION_DIR / "manual_annotation_sheet.csv"
MANUAL_ANNOTATION_SHEET_REPORT_PATH = (
    MANUAL_VALIDATION_DIR / "manual_annotation_sheet_report.txt"
)
PATIENT_VALIDATION_COHORT_PATH = MANUAL_VALIDATION_DIR / "patient_validation_cohort.csv"
PATIENT_VALIDATION_COHORT_REPORT_PATH = (
    MANUAL_VALIDATION_DIR / "patient_validation_cohort_report.txt"
)
MANUAL_VALIDATION_EVAL_DIR = MANUAL_VALIDATION_DIR / "evaluation"
MANUAL_REPORT_LABELS_PATH = MANUAL_VALIDATION_DIR / "manual_report_labels.csv"
FROZEN_VALIDATION_COHORT_DIR = MANUAL_VALIDATION_DIR / "frozen_validation_cohort"
FROZEN_PATIENT_VALIDATION_COHORT_PATH = (
    FROZEN_VALIDATION_COHORT_DIR / "patient_validation_cohort_frozen.csv"
)
FROZEN_MANUAL_REPORT_LABELS_PATH = (
    FROZEN_VALIDATION_COHORT_DIR / "manual_report_labels_frozen.csv"
)
FROZEN_COHORT_METADATA_PATH = FROZEN_VALIDATION_COHORT_DIR / "frozen_cohort_metadata.json"

# Presentation slide examples (report flow: excerpt → keywords → evidence → LLM → prediction)
PRESENTATION_EXAMPLES_DIR = ANALYSIS_DIR / "presentation_examples"
PRESENTATION_EXAMPLES_CSV_PATH = PRESENTATION_EXAMPLES_DIR / "presentation_examples.csv"
PRESENTATION_EXAMPLES_MD_PATH = PRESENTATION_EXAMPLES_DIR / "presentation_examples.md"
PRESENTATION_EXAMPLES_REPORT_PATH = (
    PRESENTATION_EXAMPLES_DIR / "presentation_examples_report.txt"
)

# Keyword / term association with predictions and baselines
KEYWORD_ANALYSIS_DIR = ANALYSIS_DIR / "keyword_analysis"
KEYWORD_ANALYSIS_TABLES_DIR = KEYWORD_ANALYSIS_DIR / "tables"
KEYWORD_ANALYSIS_PLOTS_DIR = KEYWORD_ANALYSIS_DIR / "plots"

# Field-level signal analysis (Berichte fields vs model / baselines)
FIELD_SIGNAL_ANALYSIS_DIR = ANALYSIS_DIR / "field_signal_analysis"
FIELD_SIGNAL_TABLES_DIR = FIELD_SIGNAL_ANALYSIS_DIR / "tables"
FIELD_SIGNAL_PLOTS_DIR = FIELD_SIGNAL_ANALYSIS_DIR / "plots"

# Evidence snippets (interpretability export; does not change predictions)
EVIDENCE_SNIPPETS_DIR = ANALYSIS_DIR / "evidence"
EVIDENCE_SNIPPETS_TABLES_DIR = EVIDENCE_SNIPPETS_DIR / "tables"

# --- Hemorrhage case-centric (Phase 0) ---
# Flat report file: one row per report document (semicolon CSV).
# Default: data/raw/reports.csv — override with FLAT_REPORTS_INPUT_PATH env.
_FLAT_REPORTS_ENV = os.environ.get("FLAT_REPORTS_INPUT_PATH", "").strip()
FLAT_REPORTS_INPUT_PATH = (
    Path(_FLAT_REPORTS_ENV) if _FLAT_REPORTS_ENV else REAL_RAW_DIR / "reports.csv"
)
CASES_DIR = PREPARED_DATA_DIR / "cases"
CASES_EXPORT_PATH = CASES_DIR / "clinical_cases.csv"
CASES_CONSTRUCTION_REPORT_PATH = CASES_DIR / "case_construction_report.txt"
CASE_PREDICTIONS_DIR = PREDICTIONS_DIR / "cases"
CASE_PREDICTIONS_PATH = CASE_PREDICTIONS_DIR / "case_predictions.csv"

# --- Hemorrhage real-data inspection (Excel under data/raw/) ---
INSPECTION_DIR = DATA_DIR / "inspection"
INSPECTION_RAW_SCHEMA_PATH = INSPECTION_DIR / "raw_schema_summary.csv"
INSPECTION_COLUMN_MAPPING_PATH = INSPECTION_DIR / "column_mapping_detected.csv"
INSPECTION_CASE_SUMMARY_PATH = INSPECTION_DIR / "case_summary.csv"
INSPECTION_MERGE_VALIDATION_PATH = INSPECTION_DIR / "merge_validation.csv"
INSPECTION_INCOMPLETE_CASES_PATH = INSPECTION_DIR / "incomplete_cases.csv"
INSPECTION_DUPLICATE_CASES_PATH = INSPECTION_DIR / "duplicate_cases.csv"
INSPECTION_REPORT_TYPE_DISTRIBUTION_PATH = INSPECTION_DIR / "report_type_distribution.csv"
INSPECTION_TEXT_FIELD_STATISTICS_PATH = INSPECTION_DIR / "text_field_statistics.csv"
INSPECTION_KEYWORD_EXPLORATION_PATH = INSPECTION_DIR / "keyword_exploration.csv"
INSPECTION_STRUCTURED_CASE_SAMPLES_PATH = INSPECTION_DIR / "structured_case_samples.csv"
INSPECTION_UNMATCHED_REFERENCE_PATH = INSPECTION_DIR / "unmatched_reference_rows.csv"
INSPECTION_UNMATCHED_REPORTS_PATH = INSPECTION_DIR / "unmatched_report_rows.csv"
INSPECTION_DUPLICATE_LINKAGE_PATH = INSPECTION_DIR / "duplicate_linkage_cases.csv"
INSPECTION_ANOMALY_REPORT_PATH = INSPECTION_DIR / "inspection_anomalies.txt"
INSPECTION_SUMMARY_REPORT_PATH = INSPECTION_DIR / "inspection_summary.txt"

# Reference label analytics (CCM DAVF spreadsheet)
REFERENCE_LABEL_SUMMARY_PATH = INSPECTION_DIR / "reference_label_summary.csv"
REFERENCE_LABEL_INCONSISTENCIES_PATH = INSPECTION_DIR / "reference_label_inconsistencies.csv"
REFERENCE_VALUE_DISTRIBUTION_PATH = INSPECTION_DIR / "reference_value_distribution.csv"
REFERENCE_KEYWORD_BY_LABEL_PATH = INSPECTION_DIR / "reference_keyword_by_label.csv"
REFERENCE_LABEL_ANALYSIS_SUMMARY_PATH = INSPECTION_DIR / "reference_label_analysis_summary.txt"

# Hemorrhage case-level predictions (prototype)
DATA_OUTPUTS_DIR = DATA_DIR / "outputs"
HEMORRHAGE_CASE_PREDICTIONS_PATH = DATA_OUTPUTS_DIR / "hemorrhage_case_predictions.csv"
HEMORRHAGE_PREDICTION_REVIEW_PATH = DATA_OUTPUTS_DIR / "hemorrhage_prediction_review.csv"
HEMORRHAGE_CONFUSION_REVIEW_PATH = DATA_OUTPUTS_DIR / "hemorrhage_confusion_review.csv"
HEMORRHAGE_PREDICTION_REVIEW_SUMMARY_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_prediction_review_summary.txt"
)
HEMORRHAGE_PARSE_FAILURES_PATH = DATA_OUTPUTS_DIR / "hemorrhage_parse_failures.csv"
HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_false_negative_review.csv"
)
HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_false_positive_review.csv"
)
# Final-target review exports (split of hemorrhagic predictions by clinical relevance)
HEMORRHAGE_CLINICALLY_RELEVANT_CASES_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_clinically_relevant_cases.csv"
)
HEMORRHAGE_HISTORICAL_CASES_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_historical_cases.csv"
)
HEMORRHAGE_FINAL_TARGET_SUMMARY_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_final_target_summary.csv"
)

# Classification merge: fill a patient/case spreadsheet (one row per report) with
# one-hot final-class columns. Template lives under data/raw/ (configurable via
# config.configured_classification_template_xlsx_path); merged copy is written to
# data/outputs/ so the raw template is never mutated.
HEMORRHAGE_CLASSIFICATION_MERGED_PATH = (
    DATA_OUTPUTS_DIR / "NCH_cavernom_eingeblutet_classified.xlsx"
)
HEMORRHAGE_CLASSIFICATION_MERGE_SUMMARY_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_classification_merge_summary.txt"
)
HEMORRHAGE_CLASSIFICATION_UNMATCHED_PATH = (
    DATA_OUTPUTS_DIR / "hemorrhage_classification_unmatched_rows.csv"
)

# Hemorrhage preliminary evaluation (labeled subset)
HEMORRHAGE_EVALUATION_DIR = DATA_DIR / "evaluation"
HEMORRHAGE_EVALUATION_PLOTS_DIR = HEMORRHAGE_EVALUATION_DIR / "plots"
HEMORRHAGE_METRICS_SUMMARY_CSV_PATH = HEMORRHAGE_EVALUATION_DIR / "hemorrhage_metrics_summary.csv"
HEMORRHAGE_METRICS_SUMMARY_TXT_PATH = HEMORRHAGE_EVALUATION_DIR / "hemorrhage_metrics_summary.txt"
HEMORRHAGE_METRICS_SUMMARY_MD_PATH = HEMORRHAGE_EVALUATION_DIR / "hemorrhage_metrics_summary.md"
HEMORRHAGE_EVAL_CONFUSION_MATRIX_PATH = (
    HEMORRHAGE_EVALUATION_DIR / "hemorrhage_confusion_matrix.csv"
)
HEMORRHAGE_ERROR_CASES_PATH = HEMORRHAGE_EVALUATION_DIR / "hemorrhage_error_cases.csv"
HEMORRHAGE_METRICS_SUMMARY_VERIFY_AS_NEGATIVE_CSV_PATH = (
    HEMORRHAGE_EVALUATION_DIR / "hemorrhage_metrics_summary_verify_as_negative.csv"
)
HEMORRHAGE_METRICS_SUMMARY_VERIFY_AS_NEGATIVE_TXT_PATH = (
    HEMORRHAGE_EVALUATION_DIR / "hemorrhage_metrics_summary_verify_as_negative.txt"
)
HEMORRHAGE_EVAL_CONFUSION_MATRIX_VERIFY_AS_NEGATIVE_PATH = (
    HEMORRHAGE_EVALUATION_DIR / "hemorrhage_confusion_matrix_verify_as_negative.csv"
)
