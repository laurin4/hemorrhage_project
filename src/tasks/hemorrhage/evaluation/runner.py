"""
Preliminary evaluation of hemorrhage case-level predictions on labeled subset.

Uses prediction review exports — NOT final validation until Verify_Vaskulär meaning
is clarified. Verify_Vaskulär-only cases are excluded from default performance metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.pipeline.paths import (
    HEMORRHAGE_CONFUSION_REVIEW_PATH,
    HEMORRHAGE_EVALUATION_DIR,
    HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH,
    HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH,
    HEMORRHAGE_PREDICTION_REVIEW_PATH,
)
from src.tasks.hemorrhage.evaluation.report_format import (
    SUBTYPE_ORDER,
    EvaluationReportPaths,
    build_readable_reports,
    format_pct,
)
from src.tasks.hemorrhage.export.prediction_review import compute_prediction_vs_reference

LABELED_REFERENCE_STATUSES = frozenset({"hemorrhagic", "non_hemorrhagic"})
EVALUATED_PVR_VALUES = frozenset({"TP", "TN", "FP", "FN"})

SUBTYPE_DISTRIBUTION_COLUMNS = ["haemorrhage_subtype", "count"]
SUBTYPE_BY_REFERENCE_COLUMNS = ["reference_status", "haemorrhage_subtype", "count"]

ERROR_CASE_COLUMNS: List[str] = [
    "case_id",
    "excel_pid",
    "excel_opdat",
    "reference_status",
    "klasse",
    "label",
    "prediction_vs_reference",
    "error_type",
    "sicherheit",
    "begruendung",
    "evidence_summary",
]

CONFUSION_MATRIX_COLUMNS = ["actual", "predicted", "count"]

METRIC_KEYS: List[str] = [
    "total_cases",
    "labeled_cases",
    "evaluated_cases",
    "excluded_verify_only",
    "excluded_unknown",
    "excluded_inconsistent",
    "excluded_prediction_missing",
    "parse_failed",
    "llm_failed",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "NPV",
    "F1",
    "balanced_accuracy",
]


@dataclass
class EvaluationResult:
    output_dir: Path
    plots_dir: Path
    metrics_csv_path: Path
    metrics_txt_path: Path
    metrics_md_path: Path
    confusion_matrix_path: Path
    error_cases_path: Path
    subtype_distribution_path: Optional[Path] = None
    subtype_by_reference_path: Optional[Path] = None
    summary_lines: List[str] = field(default_factory=list)
    sensitivity_summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _parse_klasse(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        k = int(float(s))
        return k if k in (0, 1) else None
    except (TypeError, ValueError):
        return None


def _row_pvr(row: pd.Series, *, include_verify_as_negative: bool) -> str:
    ref_status = str(row.get("reference_status", "") or "").strip()
    pred_status = str(row.get("status", "") or "").strip()
    klasse = _parse_klasse(row.get("klasse"))
    label = str(row.get("label", "") or "").strip()

    if include_verify_as_negative and ref_status == "verify_only":
        return compute_prediction_vs_reference(
            "non_hemorrhagic", pred_status, klasse, label
        )
    return str(row.get("prediction_vs_reference", "") or "").strip()


def _is_evaluated_row(row: pd.Series, *, include_verify_as_negative: bool) -> bool:
    ref_status = str(row.get("reference_status", "") or "").strip()
    pvr = _row_pvr(row, include_verify_as_negative=include_verify_as_negative)

    if include_verify_as_negative:
        if ref_status in LABELED_REFERENCE_STATUSES or ref_status == "verify_only":
            return pvr in EVALUATED_PVR_VALUES
        return False

    return ref_status in LABELED_REFERENCE_STATUSES and pvr in EVALUATED_PVR_VALUES


def compute_counts(review_df: pd.DataFrame, *, include_verify_as_negative: bool = False) -> Dict[str, int]:
    total = len(review_df)
    ref_status = review_df.get("reference_status", pd.Series(dtype=str)).astype(str)

    labeled = int(ref_status.isin(LABELED_REFERENCE_STATUSES).sum())
    verify_only = int((ref_status == "verify_only").sum())
    unknown = int((ref_status == "unknown").sum())
    inconsistent = int((ref_status == "inconsistent").sum())

    if "status" in review_df.columns:
        status_col = review_df["status"].astype(str)
        parse_failed = int((status_col == "parse_failed").sum())
        llm_failed = int((status_col == "llm_failed").sum())
    else:
        parse_failed = 0
        llm_failed = 0

    evaluated_mask = review_df.apply(
        lambda r: _is_evaluated_row(r, include_verify_as_negative=include_verify_as_negative),
        axis=1,
    )
    evaluated_df = review_df[evaluated_mask]
    evaluated_cases = len(evaluated_df)

    pvr_series = evaluated_df.apply(
        lambda r: _row_pvr(r, include_verify_as_negative=include_verify_as_negative),
        axis=1,
    )
    tp = int((pvr_series == "TP").sum())
    tn = int((pvr_series == "TN").sum())
    fp = int((pvr_series == "FP").sum())
    fn = int((pvr_series == "FN").sum())

    labeled_subset = review_df[ref_status.isin(LABELED_REFERENCE_STATUSES)]
    if not labeled_subset.empty and "prediction_vs_reference" in labeled_subset.columns:
        pvr_labeled = labeled_subset["prediction_vs_reference"].astype(str)
        prediction_missing = int((pvr_labeled == "prediction_missing").sum())
    else:
        prediction_missing = 0

    return {
        "total_cases": total,
        "labeled_cases": labeled,
        "evaluated_cases": evaluated_cases,
        "excluded_verify_only": verify_only if not include_verify_as_negative else 0,
        "excluded_unknown": unknown,
        "excluded_inconsistent": inconsistent,
        "excluded_prediction_missing": prediction_missing,
        "parse_failed": parse_failed,
        "llm_failed": llm_failed,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def compute_binary_metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)

    if precision is not None and sensitivity is not None:
        f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
    else:
        f1 = None

    if sensitivity is not None and specificity is not None:
        balanced_accuracy = (sensitivity + specificity) / 2.0
    else:
        balanced_accuracy = None

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "NPV": npv,
        "F1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def build_metrics_record(
    review_df: pd.DataFrame,
    *,
    include_verify_as_negative: bool = False,
) -> Dict[str, Any]:
    counts = compute_counts(review_df, include_verify_as_negative=include_verify_as_negative)
    metrics = compute_binary_metrics(counts)
    record: Dict[str, Any] = {**counts}
    for key, value in metrics.items():
        record[key] = value if value is None else round(value, 6)
    record["include_verify_as_negative"] = include_verify_as_negative
    return record


def build_confusion_matrix_rows(
    review_df: pd.DataFrame,
    *,
    include_verify_as_negative: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    evaluated = review_df[
        review_df.apply(
            lambda r: _is_evaluated_row(r, include_verify_as_negative=include_verify_as_negative),
            axis=1,
        )
    ]
    for _, row in evaluated.iterrows():
        pvr = _row_pvr(row, include_verify_as_negative=include_verify_as_negative)
        ref_status = str(row.get("reference_status", "") or "").strip()
        if include_verify_as_negative and ref_status == "verify_only":
            actual = "non_hemorrhagic (verify_as_negative)"
        elif ref_status == "hemorrhagic":
            actual = "hemorrhagic"
        elif ref_status == "non_hemorrhagic":
            actual = "non_hemorrhagic"
        else:
            actual = ref_status

        if pvr in ("TP", "FN"):
            predicted = "hemorrhagic"
        elif pvr in ("TN", "FP"):
            predicted = "non_hemorrhagic"
        else:
            continue

        rows.append({"actual": actual, "predicted": predicted, "count": 1})

    if not rows:
        return [
            {"actual": "hemorrhagic", "predicted": "hemorrhagic", "count": 0},
            {"actual": "hemorrhagic", "predicted": "non_hemorrhagic", "count": 0},
            {"actual": "non_hemorrhagic", "predicted": "hemorrhagic", "count": 0},
            {"actual": "non_hemorrhagic", "predicted": "non_hemorrhagic", "count": 0},
        ]

    grouped = (
        pd.DataFrame(rows)
        .groupby(["actual", "predicted"], as_index=False)["count"]
        .sum()
    )
    return grouped.to_dict(orient="records")


def build_error_cases_df(review_df: pd.DataFrame) -> pd.DataFrame:
    """FP/FN and labeled cases with missing predictions."""
    rows: List[Dict[str, Any]] = []
    for _, row in review_df.iterrows():
        ref_status = str(row.get("reference_status", "") or "").strip()
        if ref_status not in LABELED_REFERENCE_STATUSES:
            continue
        pvr = str(row.get("prediction_vs_reference", "") or "").strip()
        if pvr not in ("FP", "FN", "prediction_missing"):
            continue
        error_type = str(row.get("error_type", "") or "").strip()
        if not error_type:
            if pvr == "FP":
                error_type = "false_positive"
            elif pvr == "FN":
                error_type = "false_negative"
            else:
                error_type = "pipeline_failure"
        rows.append(
            {
                "case_id": row.get("case_id", ""),
                "excel_pid": row.get("excel_pid", ""),
                "excel_opdat": row.get("excel_opdat", ""),
                "reference_status": ref_status,
                "klasse": row.get("klasse", ""),
                "label": row.get("label", ""),
                "prediction_vs_reference": pvr,
                "error_type": error_type,
                "sicherheit": row.get("sicherheit", ""),
                "begruendung": row.get("begruendung", ""),
                "evidence_summary": row.get("evidence_summary", ""),
            }
        )
    return pd.DataFrame(rows, columns=ERROR_CASE_COLUMNS)


def _hemorrhagic_mask(review_df: pd.DataFrame) -> pd.Series:
    """Rows predicted hämorrhagisch (klasse==1 or label hämorrhagisch)."""
    if review_df.empty:
        return pd.Series(dtype=bool)
    label = review_df.get("label", pd.Series("", index=review_df.index)).astype(str)
    klasse = review_df.get("klasse", pd.Series("", index=review_df.index)).apply(_parse_klasse)
    return (label == "hämorrhagisch") | (klasse == 1)


def _normalized_subtype_series(review_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Predicted subtype for hämorrhagisch rows, blanks mapped to 'unbekannt'."""
    subset = review_df[mask]
    raw = subset.get(
        "predicted_haemorrhage_subtype", pd.Series("", index=subset.index)
    ).astype(str).str.strip()
    raw = raw.replace({"": "unbekannt", "nan": "unbekannt", "none": "unbekannt", "null": "unbekannt"})
    return raw


def compute_subtype_counts(review_df: pd.DataFrame) -> Dict[str, int]:
    """Predicted subtype counts among hämorrhagisch predictions (descriptive only)."""
    counts: Dict[str, int] = {k: 0 for k in SUBTYPE_ORDER}
    if review_df.empty:
        return counts
    mask = _hemorrhagic_mask(review_df)
    subtypes = _normalized_subtype_series(review_df, mask)
    for value, count in subtypes.value_counts().items():
        key = value if value in counts else "unbekannt"
        counts[key] = counts.get(key, 0) + int(count)
    return counts


def build_subtype_distribution_rows(review_df: pd.DataFrame) -> List[Dict[str, Any]]:
    counts = compute_subtype_counts(review_df)
    return [{"haemorrhage_subtype": k, "count": counts.get(k, 0)} for k in SUBTYPE_ORDER]


def build_subtype_by_reference_rows(review_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cross-tab of predicted subtype vs reference_status among hämorrhagisch predictions."""
    if review_df.empty:
        return []
    mask = _hemorrhagic_mask(review_df)
    subset = review_df[mask].copy()
    if subset.empty:
        return []
    subset["__subtype"] = _normalized_subtype_series(review_df, mask).values
    subset["__ref"] = subset.get(
        "reference_status", pd.Series("", index=subset.index)
    ).astype(str).replace({"": "unknown", "nan": "unknown"})
    grouped = (
        subset.groupby(["__ref", "__subtype"]).size().reset_index(name="count")
    )
    return [
        {
            "reference_status": r["__ref"],
            "haemorrhage_subtype": r["__subtype"],
            "count": int(r["count"]),
        }
        for _, r in grouped.iterrows()
    ]


def _format_metric_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _report_paths(
    out_dir: Path,
    *,
    review_path: Path,
    confusion_path: Path,
) -> EvaluationReportPaths:
    return EvaluationReportPaths(
        review_csv=review_path,
        confusion_review_csv=confusion_path,
        false_negative_review_csv=HEMORRHAGE_FALSE_NEGATIVE_REVIEW_PATH,
        false_positive_review_csv=HEMORRHAGE_FALSE_POSITIVE_REVIEW_PATH,
        metrics_csv=out_dir / "hemorrhage_metrics_summary.csv",
        confusion_matrix_csv=out_dir / "hemorrhage_confusion_matrix.csv",
        error_cases_csv=out_dir / "hemorrhage_error_cases.csv",
        plots_dir=out_dir / "plots",
    )


def _write_readable_reports(
    metrics: Dict[str, Any],
    paths: EvaluationReportPaths,
    txt_path: Path,
    md_path: Path,
    *,
    include_verify_as_negative: bool = False,
    subtype_counts: Optional[Dict[str, int]] = None,
) -> None:
    txt_report, md_report = build_readable_reports(
        metrics,
        paths,
        include_verify_as_negative=include_verify_as_negative,
        subtype_counts=subtype_counts,
    )
    txt_path.write_text(txt_report, encoding="utf-8")
    md_path.write_text(md_report, encoding="utf-8")


def build_console_summary_lines(
    metrics: Dict[str, Any],
    *,
    txt_path: Path,
    md_path: Path,
    include_verify_as_negative: bool = False,
) -> List[str]:
    """Short terminal summary pointing to full readable reports."""
    mode = "sensitivity" if include_verify_as_negative else "default"
    return [
        f"Hemorrhage evaluation ({mode}) — preliminary evaluation on labeled subset",
        f"evaluated_cases={_metric_int(metrics, 'evaluated_cases')} "
        f"TP={_metric_int(metrics, 'TP')} TN={_metric_int(metrics, 'TN')} "
        f"FP={_metric_int(metrics, 'FP')} FN={_metric_int(metrics, 'FN')}",
        f"accuracy={format_pct(metrics.get('accuracy'))} "
        f"sensitivity={format_pct(metrics.get('sensitivity'))} "
        f"F1={format_pct(metrics.get('F1'))}",
        f"readable_txt={txt_path}",
        f"readable_md={md_path}",
    ]


def _metric_int(metrics: Dict[str, Any], key: str) -> int:
    val = metrics.get(key)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def build_summary_lines(
    metrics: Dict[str, Any],
    *,
    include_verify_as_negative: bool = False,
) -> List[str]:
    title = (
        "Hemorrhage evaluation — sensitivity analysis (verify_only as non_hemorrhagic)"
        if include_verify_as_negative
        else "Hemorrhage evaluation — preliminary evaluation on labeled subset"
    )
    lines = [
        title,
        "(Preliminary evaluation — NOT final validation; Verify_Vaskulär meaning not yet clarified)",
        "",
        "Methodology:",
        "- Performance metrics computed only on binary labeled cases (hemorrhagic / non_hemorrhagic)",
        "- Verify_Vaskulär-only cases excluded from default performance metrics",
        "- parse_failed / llm_failed / prediction_missing excluded from performance metrics",
        "- unknown / inconsistent reference labels excluded from performance metrics",
    ]
    if include_verify_as_negative:
        lines.extend(
            [
                "",
                "SENSITIVITY ANALYSIS MODE:",
                "- verify_only cases treated as non_hemorrhagic reference (exploratory only)",
            ]
        )
    lines.append("")
    for key in METRIC_KEYS:
        lines.append(f"{key}={_format_metric_value(metrics.get(key))}")
    return lines


def _init_matplotlib():
    import matplotlib

    mpl_config = Path(__file__).resolve().parents[4] / "outputs" / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def _add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_confusion_matrix(counts: Dict[str, int], out_path: Path, *, title_suffix: str = "") -> None:
    plt, np = _init_matplotlib()
    cm = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    title = "Confusion matrix (evaluated labeled cases)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["pred non_hemorrhagic", "pred hemorrhagic"],
        yticklabels=["ref non_hemorrhagic", "ref hemorrhagic"],
        xlabel="Predicted",
        ylabel="Reference",
        title=title,
    )
    threshold = cm.max() / 2.0 if cm.max() else 0
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{int(cm[i, j])}",
                ha="center",
                va="center",
                color=color,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reference_status_distribution(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, _ = _init_matplotlib()
    counts = review_df["reference_status"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(counts.index, counts.values, color="#64748b")
    ax.set_title("Reference status distribution")
    ax.set_xlabel("reference_status")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=25)
    _add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_prediction_distribution(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, _ = _init_matplotlib()
    labels = review_df.get("label", pd.Series(dtype=str)).astype(str).replace({"": "missing"})
    labels = labels.replace("nan", "missing")
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(counts.index, counts.values, color="#3b82f6")
    ax.set_title("Predicted label distribution (all cases)")
    ax.set_xlabel("label")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=20)
    _add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_type_distribution(confusion_df: pd.DataFrame, out_path: Path) -> None:
    plt, _ = _init_matplotlib()
    if confusion_df.empty or "error_type" not in confusion_df.columns:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No confusion review data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    counts = confusion_df["error_type"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(counts.index, counts.values, color="#ef4444")
    ax.set_title("Error type distribution (confusion review)")
    ax.set_xlabel("error_type")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=30)
    _add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confidence_by_correctness(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, np = _init_matplotlib()
    evaluated = review_df[
        review_df.apply(lambda r: _is_evaluated_row(r, include_verify_as_negative=False), axis=1)
    ].copy()
    if evaluated.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No evaluated labeled cases", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    evaluated["correctness"] = evaluated["prediction_vs_reference"].map(
        {"TP": "correct", "TN": "correct", "FP": "incorrect", "FN": "incorrect"}
    )
    evaluated["sicherheit"] = evaluated["sicherheit"].astype(str).replace({"": "unknown", "nan": "unknown"})
    pivot = (
        evaluated.groupby(["sicherheit", "correctness"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["correct", "incorrect"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.35
    correct_vals = pivot.get("correct", pd.Series(0, index=pivot.index)).values
    incorrect_vals = pivot.get("incorrect", pd.Series(0, index=pivot.index)).values
    ax.bar(x - width / 2, correct_vals, width, label="correct", color="#22c55e")
    ax.bar(x + width / 2, incorrect_vals, width, label="incorrect", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_title("Confidence (sicherheit) by prediction correctness")
    ax.set_xlabel("sicherheit")
    ax.set_ylabel("Case count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_evaluated_vs_excluded(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, _ = _init_matplotlib()
    ref_status = review_df["reference_status"].astype(str)
    pvr = review_df.get("prediction_vs_reference", pd.Series(dtype=str)).astype(str)

    evaluated = int(
        review_df.apply(lambda r: _is_evaluated_row(r, include_verify_as_negative=False), axis=1).sum()
    )
    excluded_verify = int((ref_status == "verify_only").sum())
    excluded_unknown = int((ref_status == "unknown").sum())
    excluded_inconsistent = int((ref_status == "inconsistent").sum())
    labeled = review_df[ref_status.isin(LABELED_REFERENCE_STATUSES)]
    excluded_missing = int((labeled["prediction_vs_reference"].astype(str) == "prediction_missing").sum()) if not labeled.empty else 0

    categories = [
        "evaluated",
        "verify_only",
        "unknown",
        "inconsistent",
        "prediction_missing",
    ]
    values = [
        evaluated,
        excluded_verify,
        excluded_unknown,
        excluded_inconsistent,
        excluded_missing,
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(categories, values, color=["#22c55e", "#94a3b8", "#94a3b8", "#94a3b8", "#f97316"])
    ax.set_title("Evaluated vs excluded cases")
    ax.set_xlabel("Category")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=20)
    _add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_subtype_distribution(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, _ = _init_matplotlib()
    counts = compute_subtype_counts(review_df)
    labels = SUBTYPE_ORDER
    values = [counts.get(k, 0) for k in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bars = ax.bar(labels, values, color="#8b5cf6")
    ax.set_title("Predicted haemorrhage subtype distribution\n(hämorrhagisch predictions, descriptive only)")
    ax.set_xlabel("haemorrhage_subtype")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=15)
    _add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_subtype_by_reference_status(review_df: pd.DataFrame, out_path: Path) -> None:
    plt, np = _init_matplotlib()
    rows = build_subtype_by_reference_rows(review_df)
    if not rows:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(0.5, 0.5, "No hämorrhagisch predictions", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    df = pd.DataFrame(rows)
    pivot = (
        df.pivot_table(
            index="reference_status",
            columns="haemorrhage_subtype",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SUBTYPE_ORDER, fill_value=0)
    )
    ref_statuses = list(pivot.index)
    x = np.arange(len(ref_statuses))
    n_sub = len(SUBTYPE_ORDER)
    width = 0.8 / n_sub
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, subtype in enumerate(SUBTYPE_ORDER):
        vals = pivot[subtype].values
        ax.bar(x + (i - (n_sub - 1) / 2) * width, vals, width, label=subtype, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(ref_statuses, rotation=20)
    ax.set_title("Predicted subtype by reference_status\n(hämorrhagisch predictions, descriptive only)")
    ax.set_xlabel("reference_status")
    ax.set_ylabel("Case count")
    ax.legend(title="subtype", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_plots(
    review_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    counts: Dict[str, int],
    plots_dir: Path,
    *,
    include_verify_as_negative: bool = False,
) -> Tuple[List[Path], List[str]]:
    """Return (written plot paths, warnings)."""
    warnings: List[str] = []
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plots_dir / "confusion_matrix.png",
        plots_dir / "reference_status_distribution.png",
        plots_dir / "prediction_distribution.png",
        plots_dir / "error_type_distribution.png",
        plots_dir / "confidence_by_correctness.png",
        plots_dir / "evaluated_vs_excluded_cases.png",
        plots_dir / "predicted_haemorrhage_subtype_distribution.png",
        plots_dir / "subtype_by_reference_status.png",
    ]

    try:
        title_suffix = (
            "Sensitivity: verify_only treated as non_hemorrhagic"
            if include_verify_as_negative
            else "Default: verify_only excluded"
        )
        plot_confusion_matrix(counts, paths[0], title_suffix=title_suffix)
        plot_reference_status_distribution(review_df, paths[1])
        plot_prediction_distribution(review_df, paths[2])
        plot_error_type_distribution(confusion_df, paths[3])
        plot_confidence_by_correctness(review_df, paths[4])
        plot_evaluated_vs_excluded(review_df, paths[5])
        plot_subtype_distribution(review_df, paths[6])
        plot_subtype_by_reference_status(review_df, paths[7])
    except ImportError as exc:
        warnings.append(f"Plots skipped (matplotlib unavailable): {exc}")
        return [], warnings

    return paths, warnings


def run_evaluate_predictions(
    *,
    review_path: Optional[Path] = None,
    confusion_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    include_verify_as_negative: bool = False,
) -> EvaluationResult:
    rev_path = review_path or HEMORRHAGE_PREDICTION_REVIEW_PATH
    conf_path = confusion_path or HEMORRHAGE_CONFUSION_REVIEW_PATH
    out_dir = output_dir or HEMORRHAGE_EVALUATION_DIR
    plots_dir = out_dir / "plots"

    metrics_txt = out_dir / "hemorrhage_metrics_summary.txt"
    metrics_md = out_dir / "hemorrhage_metrics_summary.md"
    result = EvaluationResult(
        output_dir=out_dir,
        plots_dir=plots_dir,
        metrics_csv_path=out_dir / "hemorrhage_metrics_summary.csv",
        metrics_txt_path=metrics_txt,
        metrics_md_path=metrics_md,
        confusion_matrix_path=out_dir / "hemorrhage_confusion_matrix.csv",
        error_cases_path=out_dir / "hemorrhage_error_cases.csv",
    )

    if not rev_path.exists():
        result.errors.append(f"Review CSV missing: {rev_path}")
        result.summary_lines = [
            "No review data to evaluate.",
            "Run: python3 -m src.tasks.hemorrhage.build_prediction_review",
            *result.errors,
        ]
        return result

    review_df = pd.read_csv(rev_path)
    if review_df.empty:
        result.errors.append("Review CSV is empty")
        result.summary_lines = ["No review data to evaluate.", *result.errors]
        return result

    confusion_df = pd.read_csv(conf_path) if conf_path.exists() else pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics_record(review_df, include_verify_as_negative=False)
    subtype_counts = compute_subtype_counts(review_df)
    report_paths = _report_paths(out_dir, review_path=rev_path, confusion_path=conf_path)
    _write_readable_reports(
        metrics,
        report_paths,
        result.metrics_txt_path,
        result.metrics_md_path,
        include_verify_as_negative=False,
        subtype_counts=subtype_counts,
    )

    result.subtype_distribution_path = out_dir / "hemorrhage_subtype_distribution.csv"
    result.subtype_by_reference_path = out_dir / "hemorrhage_subtype_by_reference_status.csv"
    pd.DataFrame(
        build_subtype_distribution_rows(review_df),
        columns=SUBTYPE_DISTRIBUTION_COLUMNS,
    ).to_csv(result.subtype_distribution_path, index=False, encoding="utf-8")
    pd.DataFrame(
        build_subtype_by_reference_rows(review_df),
        columns=SUBTYPE_BY_REFERENCE_COLUMNS,
    ).to_csv(result.subtype_by_reference_path, index=False, encoding="utf-8")
    result.summary_lines = build_console_summary_lines(
        metrics,
        txt_path=result.metrics_txt_path,
        md_path=result.metrics_md_path,
    )

    pd.DataFrame([metrics]).to_csv(result.metrics_csv_path, index=False, encoding="utf-8")

    confusion_rows = build_confusion_matrix_rows(review_df, include_verify_as_negative=False)
    pd.DataFrame(confusion_rows, columns=CONFUSION_MATRIX_COLUMNS).to_csv(
        result.confusion_matrix_path, index=False, encoding="utf-8"
    )

    error_cases = build_error_cases_df(review_df)
    error_cases.to_csv(result.error_cases_path, index=False, encoding="utf-8")

    _, plot_warnings = generate_plots(
        review_df,
        confusion_df,
        metrics,
        plots_dir,
        include_verify_as_negative=False,
    )
    result.warnings.extend(plot_warnings)

    if include_verify_as_negative:
        sens_metrics = build_metrics_record(review_df, include_verify_as_negative=True)
        sens_csv = out_dir / "hemorrhage_metrics_summary_verify_as_negative.csv"
        sens_txt = out_dir / "hemorrhage_metrics_summary_verify_as_negative.txt"
        sens_md = out_dir / "hemorrhage_metrics_summary_verify_as_negative.md"
        sens_conf = out_dir / "hemorrhage_confusion_matrix_verify_as_negative.csv"

        _write_readable_reports(
            sens_metrics,
            report_paths,
            sens_txt,
            sens_md,
            include_verify_as_negative=True,
            subtype_counts=subtype_counts,
        )
        result.sensitivity_summary_lines = build_console_summary_lines(
            sens_metrics,
            txt_path=sens_txt,
            md_path=sens_md,
            include_verify_as_negative=True,
        )

        pd.DataFrame([sens_metrics]).to_csv(sens_csv, index=False, encoding="utf-8")
        pd.DataFrame(
            build_confusion_matrix_rows(review_df, include_verify_as_negative=True),
            columns=CONFUSION_MATRIX_COLUMNS,
        ).to_csv(sens_conf, index=False, encoding="utf-8")

        _, sens_plot_warnings = generate_plots(
            review_df,
            confusion_df,
            sens_metrics,
            plots_dir,
            include_verify_as_negative=True,
        )
        result.warnings.extend(sens_plot_warnings)

        result.summary_lines.extend(
            [
                "",
                f"sensitivity_metrics_csv={sens_csv}",
                f"sensitivity_metrics_txt={sens_txt}",
                f"sensitivity_metrics_md={sens_md}",
                f"sensitivity_confusion_matrix={sens_conf}",
            ]
        )

    result.summary_lines.extend(
        [
            "",
            f"metrics_csv={result.metrics_csv_path}",
            f"metrics_txt={result.metrics_txt_path}",
            f"metrics_md={result.metrics_md_path}",
            f"confusion_matrix={result.confusion_matrix_path}",
            f"error_cases={result.error_cases_path}",
            f"subtype_distribution={result.subtype_distribution_path}",
            f"subtype_by_reference_status={result.subtype_by_reference_path}",
            f"plots_dir={plots_dir}",
        ]
    )
    return result
