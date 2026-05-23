"""
Descriptive analytics for CCM DAVF reference spreadsheet labels.

No NLP, no report classification, no clinical truth inference beyond column values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.pipeline.paths import (
    INSPECTION_DIR,
    REFERENCE_KEYWORD_BY_LABEL_PATH,
    REFERENCE_LABEL_ANALYSIS_SUMMARY_PATH,
    REFERENCE_LABEL_INCONSISTENCIES_PATH,
    REFERENCE_LABEL_SUMMARY_PATH,
    REFERENCE_VALUE_DISTRIBUTION_PATH,
)
from src.tasks.hemorrhage.config import (
    REFERENCE_XLSX_ALTERNATE_FILENAMES,
    configured_reference_xlsx_path,
    reference_sheet_name,
)
from src.tasks.hemorrhage.constants import (
    CASE_KEY_ALIASES,
    REFERENCE_INDICATION_COLUMNS,
    REFERENCE_KEY_ALIASES_EXTRA,
    REFERENCE_KEYWORD_BY_LABEL_TERMS,
    REFERENCE_LABEL_COLUMN_CANDIDATES,
    REFERENCE_LABEL_TEXT_COLUMNS,
    REFERENCE_LABEL_YES_VALUES,
    REFERENCE_REQUIRED_CANONICAL_KEYS,
)
from src.tasks.hemorrhage.io.column_normalize import normalize_dataframe_columns
from src.tasks.hemorrhage.io.excel_loader import load_excel_raw
from src.tasks.hemorrhage.io.key_normalize import merge_reference_key_aliases
from src.tasks.hemorrhage.io.path_resolve import resolve_raw_input_path

LOGGER = logging.getLogger(__name__)

LabelTriState = str  # "yes" | "no" | "missing" | "non_standard"


@dataclass
class ReferenceLabelAnalysisResult:
    reference_path: Path
    total_rows: int = 0
    output_paths: List[Path] = field(default_factory=list)
    summary_lines: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _header_lookup(columns: Sequence[str]) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def resolve_reference_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    lookup = _header_lookup(df.columns)
    for cand in candidates:
        hit = lookup.get(cand.strip().lower())
        if hit is not None:
            return hit
    return None


def resolve_label_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map canonical label keys to actual dataframe column names."""
    out: Dict[str, Optional[str]] = {}
    for key, candidates in REFERENCE_LABEL_COLUMN_CANDIDATES.items():
        out[key] = resolve_reference_column(df, candidates)
    return out


def parse_label_value(raw: object) -> Tuple[LabelTriState, str]:
    """
    Parse spreadsheet label cell → (state, raw_display).

    yes: ja/yes/1/true (case-insensitive)
    missing: empty / NaN
    no: explicit 0/nein/no/false
    non_standard: anything else (preserved, not coerced)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "missing", ""
    if isinstance(raw, bool):
        return ("yes", "True") if raw else ("no", "False")
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        if raw == 1 or raw == 1.0:
            return "yes", str(int(raw))
        if raw == 0 or raw == 0.0:
            return "no", str(int(raw))

    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return "missing", ""

    low = s.lower()
    if low in REFERENCE_LABEL_YES_VALUES:
        return "yes", s
    if low in ("nein", "no", "0", "false", "n", "falsch"):
        return "no", s
    return "non_standard", s


def assign_reference_label_stratum(
    hemo: LabelTriState,
    non_hemo: LabelTriState,
) -> str:
    if hemo == "non_standard" or non_hemo == "non_standard":
        return "unclear"
    if hemo == "yes" and non_hemo == "yes":
        return "both_marked"
    if hemo == "yes" and non_hemo != "yes":
        return "hemorrhagic"
    if non_hemo == "yes" and hemo != "yes":
        return "non_hemorrhagic"
    if hemo == "missing" and non_hemo == "missing":
        return "unlabeled"
    if hemo == "no" and non_hemo == "no":
        return "neither_yes"
    return "unclear"


def load_normalized_reference(reference_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Path, List[str]]:
    """Load reference Excel with canonical merge keys normalized."""
    errors: List[str] = []
    configured = reference_path or configured_reference_xlsx_path()
    resolved = resolve_raw_input_path(
        configured, REFERENCE_XLSX_ALTERNATE_FILENAMES, context="reference_labels"
    )
    if resolved.resolution == "missing":
        errors.append(f"Reference file missing: {configured}")
        return pd.DataFrame(), resolved.resolved_path, errors

    df, load_report = load_excel_raw(
        resolved.resolved_path,
        source_label="reference_labels",
        sheet_name=reference_sheet_name(),
    )
    errors.extend(load_report.errors)

    ref_aliases = merge_reference_key_aliases(CASE_KEY_ALIASES, REFERENCE_KEY_ALIASES_EXTRA)
    df, _ = normalize_dataframe_columns(
        df,
        source_label="reference_labels",
        extra_aliases=ref_aliases,
        required_case_keys=REFERENCE_REQUIRED_CANONICAL_KEYS,
    )
    return df, resolved.resolved_path, errors


def label_balance_summary(df: pd.DataFrame, label_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    n = len(df)
    hemo_col = label_cols.get("haemorrhagisch")
    non_col = label_cols.get("nicht_haemorrhagisch")
    ver_col = label_cols.get("verify_vaskulaer")

    hemo_yes = non_yes = both_yes = neither_yes = missing_both = 0
    verify_yes = verify_missing = 0

    if hemo_col and non_col:
        for _, row in df.iterrows():
            h_state, _ = parse_label_value(row.get(hemo_col))
            n_state, _ = parse_label_value(row.get(non_col))
            if h_state == "yes":
                hemo_yes += 1
            if n_state == "yes":
                non_yes += 1
            if h_state == "yes" and n_state == "yes":
                both_yes += 1
            if h_state != "yes" and n_state != "yes":
                if h_state == "missing" and n_state == "missing":
                    missing_both += 1
                elif h_state in ("no", "missing") and n_state in ("no", "missing"):
                    neither_yes += 1

    if ver_col:
        for _, row in df.iterrows():
            v_state, _ = parse_label_value(row.get(ver_col))
            if v_state == "yes":
                verify_yes += 1
            if v_state == "missing":
                verify_missing += 1

    rows = [
        {"metric": "total_rows", "count": n},
        {"metric": "haemorrhagisch_yes", "count": hemo_yes},
        {"metric": "nicht_haemorrhagisch_yes", "count": non_yes},
        {"metric": "both_haemorrhagisch_and_nicht_yes", "count": both_yes},
        {"metric": "neither_yes", "count": neither_yes},
        {"metric": "both_missing", "count": missing_both},
        {"metric": "verify_vaskulaer_yes", "count": verify_yes},
        {"metric": "verify_vaskulaer_missing", "count": verify_missing},
    ]
    return pd.DataFrame(rows)


def label_inconsistencies_dataframe(
    df: pd.DataFrame,
    label_cols: Dict[str, Optional[str]],
) -> pd.DataFrame:
    hemo_col = label_cols.get("haemorrhagisch")
    non_col = label_cols.get("nicht_haemorrhagisch")
    ver_col = label_cols.get("verify_vaskulaer")

    rows: List[dict] = []
    for idx, row in df.iterrows():
        issues: List[str] = []
        h_state, h_raw = ("missing", "") if not hemo_col else parse_label_value(row.get(hemo_col))
        n_state, n_raw = ("missing", "") if not non_col else parse_label_value(row.get(non_col))
        v_state, v_raw = ("missing", "") if not ver_col else parse_label_value(row.get(ver_col))

        if h_state == "yes" and n_state == "yes":
            issues.append("both_haemorrhagisch_and_nicht_yes")
        if h_state != "yes" and n_state != "yes":
            if h_state == "missing" and n_state == "missing":
                issues.append("both_missing")
            elif h_state in ("no", "missing") and n_state in ("no", "missing"):
                issues.append("neither_yes")
        if h_state == "non_standard":
            issues.append("non_standard_haemorrhagisch")
        if n_state == "non_standard":
            issues.append("non_standard_nicht_haemorrhagisch")
        if ver_col and v_state == "non_standard":
            issues.append("non_standard_verify_vaskulaer")

        if not issues:
            continue

        out_row = {
            "row_index": int(idx),
            "excel_pid": row.get("excel_pid", ""),
            "excel_opdat": row.get("excel_opdat", ""),
            "issues": "|".join(issues),
            "haemorrhagisch_raw": h_raw,
            "nicht_haemorrhagisch_raw": n_raw,
            "verify_vaskulaer_raw": v_raw,
            "label_stratum": assign_reference_label_stratum(h_state, n_state),
        }
        rows.append(out_row)

    return pd.DataFrame(rows)


def _stratum_series(df: pd.DataFrame, label_cols: Dict[str, Optional[str]]) -> pd.Series:
    hemo_col = label_cols.get("haemorrhagisch")
    non_col = label_cols.get("nicht_haemorrhagisch")
    strata: List[str] = []
    for _, row in df.iterrows():
        h_state, _ = ("missing", "") if not hemo_col else parse_label_value(row.get(hemo_col))
        n_state, _ = ("missing", "") if not non_col else parse_label_value(row.get(non_col))
        strata.append(assign_reference_label_stratum(h_state, n_state))
    return pd.Series(strata, index=df.index)


def value_distribution_dataframe(
    df: pd.DataFrame,
    label_cols: Dict[str, Optional[str]],
) -> pd.DataFrame:
    strata = _stratum_series(df, label_cols)
    work = df.copy()
    work["_label_stratum"] = strata

    rows: List[dict] = []
    for col_name in REFERENCE_INDICATION_COLUMNS:
        actual = resolve_reference_column(df, (col_name,))
        if actual is None:
            rows.append(
                {
                    "column": col_name,
                    "value": "(column_missing)",
                    "count_overall": len(df),
                    "count_hemorrhagic": 0,
                    "count_non_hemorrhagic": 0,
                    "count_unlabeled_unclear": len(df),
                }
            )
            continue

        for value, grp in work.groupby(actual, dropna=False):
            val_str = "" if pd.isna(value) else str(value).strip()
            if not val_str:
                val_str = "(empty)"
            rows.append(
                {
                    "column": col_name,
                    "value": val_str,
                    "count_overall": len(grp),
                    "count_hemorrhagic": int((grp["_label_stratum"] == "hemorrhagic").sum()),
                    "count_non_hemorrhagic": int((grp["_label_stratum"] == "non_hemorrhagic").sum()),
                    "count_unlabeled_unclear": int(
                        grp["_label_stratum"].isin(
                            ("unlabeled", "unclear", "both_marked", "neither_yes")
                        ).sum()
                    ),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["column", "count_overall"], ascending=[True, False])
    return out


def keyword_by_label_dataframe(
    df: pd.DataFrame,
    label_cols: Dict[str, Optional[str]],
) -> pd.DataFrame:
    strata = _stratum_series(df, label_cols)
    work = df.copy()
    work["_label_stratum"] = strata

    rows: List[dict] = []
    for col_name in REFERENCE_LABEL_TEXT_COLUMNS:
        actual = resolve_reference_column(df, (col_name,))
        if actual is None:
            for term in REFERENCE_KEYWORD_BY_LABEL_TERMS:
                rows.append(
                    {
                        "text_column": col_name,
                        "keyword": term,
                        "column_present": False,
                        "rows_with_keyword": 0,
                        "count_hemorrhagic": 0,
                        "count_non_hemorrhagic": 0,
                        "count_unlabeled_unclear": 0,
                    }
                )
            continue

        text_series = work[actual].fillna("").astype(str)
        for term in REFERENCE_KEYWORD_BY_LABEL_TERMS:
            hits = text_series.str.lower().str.contains(term.lower(), regex=False, na=False)
            hit_df = work.loc[hits]
            rows.append(
                {
                    "text_column": col_name,
                    "keyword": term,
                    "column_present": True,
                    "rows_with_keyword": int(hits.sum()),
                    "count_hemorrhagic": int((hit_df["_label_stratum"] == "hemorrhagic").sum()),
                    "count_non_hemorrhagic": int((hit_df["_label_stratum"] == "non_hemorrhagic").sum()),
                    "count_unlabeled_unclear": int(
                        hit_df["_label_stratum"].isin(
                            ("unlabeled", "unclear", "both_marked", "neither_yes")
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_reference_label_analysis(
    *,
    reference_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ReferenceLabelAnalysisResult:
    out_dir = output_dir or INSPECTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df, path, errors = load_normalized_reference(reference_path)
    result = ReferenceLabelAnalysisResult(reference_path=path, errors=errors)

    if df.empty:
        result.summary_lines = ["Reference label analysis: no data loaded.", *errors]
        return result

    result.total_rows = len(df)
    label_cols = resolve_label_columns(df)

    missing_labels = [k for k, v in label_cols.items() if v is None]
    if missing_labels:
        result.errors.append(f"Missing label columns: {missing_labels}")

    balance = label_balance_summary(df, label_cols)
    inconsistencies = label_inconsistencies_dataframe(df, label_cols)
    distribution = value_distribution_dataframe(df, label_cols)
    keywords = keyword_by_label_dataframe(df, label_cols)

    paths = [
        (REFERENCE_LABEL_SUMMARY_PATH.name, balance),
        (REFERENCE_LABEL_INCONSISTENCIES_PATH.name, inconsistencies),
        (REFERENCE_VALUE_DISTRIBUTION_PATH.name, distribution),
        (REFERENCE_KEYWORD_BY_LABEL_PATH.name, keywords),
    ]
    for name, frame in paths:
        p = out_dir / name
        frame.to_csv(p, index=False, encoding="utf-8")
        result.output_paths.append(p)

    summary_lines = [
        "Reference label analysis (descriptive only)",
        "=" * 44,
        f"reference_path={path}",
        f"total_rows={result.total_rows}",
        "",
        "Label columns resolved:",
    ]
    for k, v in label_cols.items():
        summary_lines.append(f"  {k}: {v or '(missing)'}")
    summary_lines.append("")
    if not balance.empty:
        for _, r in balance.iterrows():
            summary_lines.append(f"  {r['metric']}={int(r['count'])}")
    summary_lines.append("")
    summary_lines.append(f"inconsistent_or_unclear_rows={len(inconsistencies)}")
    summary_lines.append("")
    summary_lines.append("Output files:")
    for p in result.output_paths:
        summary_lines.append(f"  {p}")

    result.summary_lines = summary_lines
    summary_path = out_dir / REFERENCE_LABEL_ANALYSIS_SUMMARY_PATH.name
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    result.output_paths.append(summary_path)

    return result
