"""
Map Excel headers to canonical column names with a full audit trail.

Critical identifiers (case keys) are preserved as strings; coercion issues are logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.tasks.hemorrhage.constants import CASE_KEY_ALIASES, TYPUS_ALIASES

LOGGER = logging.getLogger(__name__)

CASE_KEY_COLUMNS = ("excel_pid", "excel_opdat", "opber_fallnr")


@dataclass
class ColumnMappingReport:
    source_label: str
    mappings: List[Dict[str, str]] = field(default_factory=list)
    unmapped_columns: List[str] = field(default_factory=list)
    missing_canonical: List[str] = field(default_factory=list)
    duplicate_target_warnings: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.mappings:
            return pd.DataFrame(columns=["source_label", "original_column", "canonical_column", "status"])
        rows = [{**m, "source_label": self.source_label} for m in self.mappings]
        return pd.DataFrame(rows)


def _header_lookup(columns: Sequence[str]) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _resolve_alias(lookup: Dict[str, str], aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        hit = lookup.get(alias.lower())
        if hit is not None:
            return hit
    return None


def normalize_dataframe_columns(
    df: pd.DataFrame,
    *,
    source_label: str,
    extra_aliases: Optional[Dict[str, Tuple[str, ...]]] = None,
    typus_aliases: Sequence[str] = TYPUS_ALIASES,
) -> Tuple[pd.DataFrame, ColumnMappingReport]:
    """
    Rename columns to canonical names where aliases match.

    Does not drop unmapped columns. Does not modify cell values except case-key
    columns converted to string with ``astype(str)`` after strip (NaN preserved as empty).
    """
    report = ColumnMappingReport(source_label=source_label)
    if df.empty:
        report.missing_canonical = list(CASE_KEY_COLUMNS)
        return df.copy(), report

    lookup = _header_lookup(df.columns)
    rename_map: Dict[str, str] = {}
    targets_seen: Dict[str, str] = {}

    all_aliases: Dict[str, Tuple[str, ...]] = dict(CASE_KEY_ALIASES)
    if extra_aliases:
        all_aliases.update(extra_aliases)
    all_aliases["typus"] = tuple(typus_aliases)

    for canonical, aliases in all_aliases.items():
        original = _resolve_alias(lookup, aliases)
        if original is None:
            if canonical in CASE_KEY_COLUMNS or canonical == "typus":
                report.missing_canonical.append(canonical)
            report.mappings.append(
                {
                    "original_column": "",
                    "canonical_column": canonical,
                    "status": "missing",
                }
            )
            continue
        if canonical in targets_seen:
            report.duplicate_target_warnings.append(
                f"{canonical}: already mapped from {targets_seen[canonical]!r}, also saw {original!r}"
            )
        targets_seen[canonical] = original
        rename_map[original] = canonical
        report.mappings.append(
            {
                "original_column": original,
                "canonical_column": canonical,
                "status": "mapped",
            }
        )

    for col in df.columns:
        c = str(col)
        if c not in rename_map and c not in rename_map.values():
            report.unmapped_columns.append(c)

    out = df.rename(columns=rename_map).copy()

    for key in CASE_KEY_COLUMNS:
        if key in out.columns:
            out[key] = _identifier_series_as_string(out[key], column=key, source_label=source_label)

    if "typus" in out.columns:
        out["typus"] = out["typus"].map(_cell_as_display_string)

    return out, report


def _cell_as_display_string(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).strip()


def _identifier_series_as_string(series: pd.Series, *, column: str, source_label: str) -> pd.Series:
    """Stringify identifiers; log non-string dtypes and datetime conversions."""
    if pd.api.types.is_datetime64_any_dtype(series):
        LOGGER.warning(
            "[%s] Column %s is datetime — converting to ISO date string (review for unintended coercion).",
            source_label,
            column,
        )
        return series.dt.strftime("%Y-%m-%d").fillna("")

    non_null = series.dropna()
    if len(non_null) and not pd.api.types.is_string_dtype(series):
        sample = non_null.iloc[0]
        LOGGER.info(
            "[%s] Column %s dtype=%s — stringifying (sample=%r).",
            source_label,
            column,
            series.dtype,
            sample,
        )

    def _one(v: object) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v).strip()

    return series.map(_one)
