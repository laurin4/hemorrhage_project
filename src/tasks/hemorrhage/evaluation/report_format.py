"""
Human-readable evaluation reports (TXT / Markdown) for supervisor meetings and thesis notes.

Presentation only — does not change metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvaluationReportPaths:
    """Related artifact paths to cite in the report."""

    review_csv: Path
    confusion_review_csv: Path
    false_negative_review_csv: Path
    false_positive_review_csv: Path
    metrics_csv: Path
    confusion_matrix_csv: Path
    error_cases_csv: Path
    plots_dir: Path


SUBTYPE_ORDER: List[str] = ["akut", "nicht_akut", "historisch", "unbekannt"]

_SUBTYPE_NOTE = (
    "Subtype analysis is descriptive only. Historical hemorrhage is treated as "
    "hämorrhagisch for binary evaluation. (No validated reference subtype labels "
    "are currently available, so subtype accuracy is not computed.)"
)


def _subtype_total(subtype_counts: Optional[Dict[str, int]]) -> int:
    if not subtype_counts:
        return 0
    return sum(int(subtype_counts.get(k, 0)) for k in SUBTYPE_ORDER)


def build_subtype_section_txt(subtype_counts: Optional[Dict[str, int]]) -> List[str]:
    counts = subtype_counts or {}
    total = _subtype_total(counts)
    lines = [
        "Hemorrhage subtype analysis",
        "-" * 40,
        f"Predicted haemorrhagisch cases by subtype (total {total}):",
    ]
    for key in SUBTYPE_ORDER:
        lines.append(f"- {key}: {int(counts.get(key, 0))}")
    lines.extend(["", _SUBTYPE_NOTE])
    return lines


def build_subtype_section_md(subtype_counts: Optional[Dict[str, int]]) -> List[str]:
    counts = subtype_counts or {}
    total = _subtype_total(counts)
    lines = [
        "## Hemorrhage subtype analysis",
        "",
        f"Predicted hämorrhagisch cases by subtype (total {total}):",
        "",
        "| Subtype | Count | Share of hämorrhagisch |",
        "|---------|------:|-----------------------:|",
    ]
    for key in SUBTYPE_ORDER:
        c = int(counts.get(key, 0))
        lines.append(f"| {key} | {c} | {format_share(c, total)} |")
    lines.extend(["", f"> {_SUBTYPE_NOTE}"])
    return lines


def _metric_int(metrics: Dict[str, Any], key: str) -> int:
    val = metrics.get(key)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def format_pct(value: Optional[float], *, decimals: int = 1) -> str:
    """Format a 0–1 rate as percentage with one decimal (Unicode-safe)."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def format_share(count: int, total: int, *, decimals: int = 1) -> str:
    if total <= 0:
        return "n/a"
    return f"{count / total * 100:.{decimals}f}%"


def build_ascii_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> List[str]:
    """ASCII confusion matrix (rows = reference, cols = predicted)."""
    col_w = max(6, len(str(max(tp, tn, fp, fn, 0))) + 2)
    hdr_pred = " " * 14 + "pred non_h".ljust(col_w) + "pred haemo".ljust(col_w)
    row_non = "ref non_h".ljust(14) + str(tn).center(col_w) + str(fp).center(col_w)
    row_hae = "ref haemo".ljust(14) + str(fn).center(col_w) + str(tp).center(col_w)
    return [
        hdr_pred,
        row_non,
        row_hae,
        "",
        f"  (TN={tn}, FP={fp}, FN={fn}, TP={tp})",
    ]


def _class_balance_lines(metrics: Dict[str, Any]) -> List[str]:
    """Reference class balance among evaluated cases (hemorrhagic = TP+FN)."""
    tp = _metric_int(metrics, "TP")
    fn = _metric_int(metrics, "FN")
    tn = _metric_int(metrics, "TN")
    fp = _metric_int(metrics, "FP")
    ref_pos = tp + fn
    ref_neg = tn + fp
    evaluated = ref_pos + ref_neg
    if evaluated == 0:
        return ["Class balance (evaluated): n/a (no evaluated cases)"]
    pos_pct = ref_pos / evaluated * 100
    neg_pct = ref_neg / evaluated * 100
    lines = [
        f"Reference hemorrhagic (evaluated): {ref_pos} ({pos_pct:.1f}%)",
        f"Reference non-hemorrhagic (evaluated): {ref_neg} ({neg_pct:.1f}%)",
    ]
    if ref_pos > 0 and ref_neg > 0:
        ratio = max(ref_pos, ref_neg) / min(ref_pos, ref_neg)
        if ratio >= 2.0:
            lines.append(
                f"Note: Class imbalance among evaluated cases (ratio about {ratio:.1f}:1)."
            )
    return lines


def build_interpretation_bullets(
    metrics: Dict[str, Any],
    *,
    include_verify_as_negative: bool = False,
) -> List[str]:
    tp = _metric_int(metrics, "TP")
    tn = _metric_int(metrics, "TN")
    fp = _metric_int(metrics, "FP")
    fn = _metric_int(metrics, "FN")
    evaluated = _metric_int(metrics, "evaluated_cases")
    excluded_verify = _metric_int(metrics, "excluded_verify_only")
    sensitivity = metrics.get("sensitivity")

    bullets: List[str] = []

    if include_verify_as_negative:
        bullets.append(
            "Sensitivity analysis: verify_only cases were treated as non_hemorrhagic "
            "(exploratory only — not the default conservative methodology)."
        )
    else:
        bullets.append(
            "Verify_Vaskulaer-only cases were excluded from performance metrics "
            "(default conservative methodology)."
        )

    bullets.append(
        "Results are preliminary and depend on final clinical definition clarification "
        "(not final validation)."
    )

    if evaluated == 0:
        bullets.append("No evaluated labeled cases — metrics are not interpretable yet.")
        return bullets

    if fn > fp and fn >= tp:
        bullets.append(
            "The model appears conservative for hemorrhage detection "
            f"(FN={fn} vs FP={fp}, TP={tp})."
        )
        bullets.append(
            "High FN count may indicate an overly strict hemorrhage definition, "
            "prompt behaviour, or difficulty detecting relevant pre-operative bleeding."
        )
    elif fp > fn and fp >= tn:
        bullets.append(
            f"False positives (FP={fp}) exceed false negatives (FN={fn}) in this subset."
        )
        bullets.append(
            "Review FP cases for cavernoma / descriptive bleeding language "
            "(e.g. geblutetes Kavernom) driving hämorrhagisch predictions."
        )
    else:
        bullets.append(
            f"Error mix: FN={fn}, FP={fp} on {evaluated} evaluated labeled cases."
        )

    if sensitivity is not None and float(sensitivity) < 0.5 and (tp + fn) > 0:
        bullets.append(
            f"Sensitivity (recall) is low ({format_pct(sensitivity)}) — many reference "
            "hemorrhagic cases were missed."
        )

    if excluded_verify > 0:
        bullets.append(
            f"{excluded_verify} verify_only case(s) await label clarification before "
            "inclusion in any final evaluation."
        )

    return bullets


def _path_display(path: Path) -> str:
    return str(path)


def build_readable_report_txt(
    metrics: Dict[str, Any],
    paths: EvaluationReportPaths,
    *,
    include_verify_as_negative: bool = False,
    subtype_counts: Optional[Dict[str, int]] = None,
) -> str:
    title = (
        "Hemorrhage Evaluation — Sensitivity Analysis (verify_only as non_hemorrhagic)"
        if include_verify_as_negative
        else "Hemorrhage Preliminary Evaluation"
    )
    subtitle = (
        "Preliminary evaluation on labeled subset — NOT final validation"
    )

    total = _metric_int(metrics, "total_cases")
    labeled = _metric_int(metrics, "labeled_cases")
    evaluated = _metric_int(metrics, "evaluated_cases")
    tp, tn, fp, fn = (
        _metric_int(metrics, "TP"),
        _metric_int(metrics, "TN"),
        _metric_int(metrics, "FP"),
        _metric_int(metrics, "FN"),
    )

    sep = "=" * 66
    lines: List[str] = [
        sep,
        title,
        subtitle,
        sep,
        "",
        "Dataset overview",
        "-" * 40,
        f"Total cases:              {total}",
        f"Labeled cases:            {labeled}  ({format_share(labeled, total)} of total)",
        f"Evaluated cases:          {evaluated}  ({format_share(evaluated, total)} of total)",
        "",
        "Excluded cases",
        "-" * 40,
        f"Verify-only excluded:     {_metric_int(metrics, 'excluded_verify_only')}  "
        f"({format_share(_metric_int(metrics, 'excluded_verify_only'), total)} of total)",
        f"Unknown reference:        {_metric_int(metrics, 'excluded_unknown')}",
        f"Inconsistent reference:   {_metric_int(metrics, 'excluded_inconsistent')}",
        f"Prediction missing:       {_metric_int(metrics, 'excluded_prediction_missing')}",
        f"Parse failed (pipeline):  {_metric_int(metrics, 'parse_failed')}",
        f"LLM failed (pipeline):    {_metric_int(metrics, 'llm_failed')}",
        "",
        "Confusion matrix (evaluated labeled cases only)",
        "-" * 40,
        f"TP: {tp}",
        f"TN: {tn}",
        f"FP: {fp}",
        f"FN: {fn}",
        "",
    ]
    lines.extend(build_ascii_confusion_matrix(tp, tn, fp, fn))
    lines.extend(
        [
            "",
            "Performance metrics (labeled subset only)",
            "-" * 40,
            f"Accuracy:              {format_pct(metrics.get('accuracy'))}",
            f"Sensitivity (Recall):  {format_pct(metrics.get('sensitivity'))}",
            f"Specificity:           {format_pct(metrics.get('specificity'))}",
            f"Precision (PPV):       {format_pct(metrics.get('precision'))}",
            f"NPV:                   {format_pct(metrics.get('NPV'))}",
            f"F1 Score:              {format_pct(metrics.get('F1'))}",
            f"Balanced Accuracy:     {format_pct(metrics.get('balanced_accuracy'))}",
            "",
            "Class balance (evaluated subset)",
            "-" * 40,
        ]
    )
    lines.extend(_class_balance_lines(metrics))
    lines.extend(
        [
            "",
            "Interpretation",
            "-" * 40,
        ]
    )
    for bullet in build_interpretation_bullets(
        metrics, include_verify_as_negative=include_verify_as_negative
    ):
        lines.append(f"* {bullet}")

    lines.append("")
    lines.extend(build_subtype_section_txt(subtype_counts))

    lines.extend(
        [
            "",
            "Related outputs",
            "-" * 40,
            f"Metrics CSV:            {_path_display(paths.metrics_csv)}",
            f"Confusion matrix CSV:   {_path_display(paths.confusion_matrix_csv)}",
            f"Error cases CSV:        {_path_display(paths.error_cases_csv)}",
            f"Prediction review:      {_path_display(paths.review_csv)}",
            f"Confusion review:       {_path_display(paths.confusion_review_csv)}",
            f"FN detailed review:     {_path_display(paths.false_negative_review_csv)}",
            f"FP detailed review:     {_path_display(paths.false_positive_review_csv)}",
            f"Plots directory:        {_path_display(paths.plots_dir)}",
            "",
            sep,
        ]
    )
    return "\n".join(lines) + "\n"


def build_readable_report_md(
    metrics: Dict[str, Any],
    paths: EvaluationReportPaths,
    *,
    include_verify_as_negative: bool = False,
    subtype_counts: Optional[Dict[str, int]] = None,
) -> str:
    title = (
        "Hemorrhage Evaluation — Sensitivity Analysis"
        if include_verify_as_negative
        else "Hemorrhage Preliminary Evaluation"
    )

    total = _metric_int(metrics, "total_cases")
    labeled = _metric_int(metrics, "labeled_cases")
    evaluated = _metric_int(metrics, "evaluated_cases")
    tp, tn, fp, fn = (
        _metric_int(metrics, "TP"),
        _metric_int(metrics, "TN"),
        _metric_int(metrics, "FP"),
        _metric_int(metrics, "FN"),
    )

    lines: List[str] = [
        f"# {title}",
        "",
        "> **Preliminary evaluation on labeled subset** — not final validation. "
        "Verify_Vaskulaer meaning not yet clarified.",
        "",
    ]
    if include_verify_as_negative:
        lines.append(
            "> **Sensitivity mode:** verify_only treated as non_hemorrhagic (exploratory)."
        )
        lines.append("")

    lines.extend(
        [
            "## Dataset overview",
            "",
            "| Metric | Count | Share of total |",
            "|--------|------:|---------------:|",
            f"| Total cases | {total} | 100.0% |",
            f"| Labeled cases | {labeled} | {format_share(labeled, total)} |",
            f"| Evaluated cases | {evaluated} | {format_share(evaluated, total)} |",
            "",
            "## Excluded cases",
            "",
            "| Category | Count | Share of total |",
            "|----------|------:|---------------:|",
            f"| Verify-only excluded | {_metric_int(metrics, 'excluded_verify_only')} | "
            f"{format_share(_metric_int(metrics, 'excluded_verify_only'), total)} |",
            f"| Unknown reference | {_metric_int(metrics, 'excluded_unknown')} | "
            f"{format_share(_metric_int(metrics, 'excluded_unknown'), total)} |",
            f"| Inconsistent reference | {_metric_int(metrics, 'excluded_inconsistent')} | "
            f"{format_share(_metric_int(metrics, 'excluded_inconsistent'), total)} |",
            f"| Prediction missing | {_metric_int(metrics, 'excluded_prediction_missing')} | "
            f"{format_share(_metric_int(metrics, 'excluded_prediction_missing'), total)} |",
            f"| Parse failed | {_metric_int(metrics, 'parse_failed')} | "
            f"{format_share(_metric_int(metrics, 'parse_failed'), total)} |",
            f"| LLM failed | {_metric_int(metrics, 'llm_failed')} | "
            f"{format_share(_metric_int(metrics, 'llm_failed'), total)} |",
            "",
            "## Confusion matrix",
            "",
            "| | Predicted non-hemorrhagic | Predicted hemorrhagic |",
            "|--|--:|--:|",
            f"| **Reference non-hemorrhagic** | {tn} (TN) | {fp} (FP) |",
            f"| **Reference hemorrhagic** | {fn} (FN) | {tp} (TP) |",
            "",
            "### ASCII view",
            "",
            "```text",
            *build_ascii_confusion_matrix(tp, tn, fp, fn),
            "```",
            "",
            "## Performance metrics (labeled subset only)",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Accuracy | {format_pct(metrics.get('accuracy'))} |",
            f"| Sensitivity (Recall) | {format_pct(metrics.get('sensitivity'))} |",
            f"| Specificity | {format_pct(metrics.get('specificity'))} |",
            f"| Precision (PPV) | {format_pct(metrics.get('precision'))} |",
            f"| NPV | {format_pct(metrics.get('NPV'))} |",
            f"| F1 Score | {format_pct(metrics.get('F1'))} |",
            f"| Balanced Accuracy | {format_pct(metrics.get('balanced_accuracy'))} |",
            "",
            "## Class balance (evaluated subset)",
            "",
        ]
    )
    for line in _class_balance_lines(metrics):
        lines.append(f"- {line}")
    lines.extend(["", "## Interpretation", ""])
    for bullet in build_interpretation_bullets(
        metrics, include_verify_as_negative=include_verify_as_negative
    ):
        lines.append(f"- {bullet}")

    lines.append("")
    lines.extend(build_subtype_section_md(subtype_counts))

    lines.extend(
        [
            "",
            "## Related outputs",
            "",
            f"- Metrics CSV: `{paths.metrics_csv}`",
            f"- Confusion matrix CSV: `{paths.confusion_matrix_csv}`",
            f"- Error cases CSV: `{paths.error_cases_csv}`",
            f"- Prediction review: `{paths.review_csv}`",
            f"- Confusion review: `{paths.confusion_review_csv}`",
            f"- FN detailed review: `{paths.false_negative_review_csv}`",
            f"- FP detailed review: `{paths.false_positive_review_csv}`",
            f"- Plots: `{paths.plots_dir}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_readable_reports(
    metrics: Dict[str, Any],
    paths: EvaluationReportPaths,
    *,
    include_verify_as_negative: bool = False,
    subtype_counts: Optional[Dict[str, int]] = None,
) -> Tuple[str, str]:
    """Return (txt_report, md_report)."""
    txt = build_readable_report_txt(
        metrics,
        paths,
        include_verify_as_negative=include_verify_as_negative,
        subtype_counts=subtype_counts,
    )
    md = build_readable_report_md(
        metrics,
        paths,
        include_verify_as_negative=include_verify_as_negative,
        subtype_counts=subtype_counts,
    )
    return txt, md
