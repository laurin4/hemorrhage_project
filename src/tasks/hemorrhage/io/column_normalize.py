"""
Map Excel headers to canonical column names with a full audit trail.

Critical identifiers (case keys) are preserved as strings; coercion issues are logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.tasks.hemorrhage.constants import CASE_KEY_ALIASES, CASE_KEY_COLUMNS, TYPUS_ALIASES
from src.tasks.hemorrhage.io.key_normalize import (
    apply_canonical_merge_key_normalization,
    normalize_excel_pid_series,
)

LOGGER = logging.getLogger(__name__)


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
    required_case_keys: Sequence[str] = CASE_KEY_COLUMNS,
    normalize_merge_keys: bool = True,
) -> Tuple[pd.DataFrame, ColumnMappingReport]:
    """
    Rename columns to canonical names where aliases match.

    Does not drop unmapped columns. Applies shared ``excel_pid`` / ``excel_opdat``
    normalization when *normalize_merge_keys* is True.
    """
    report = ColumnMappingReport(source_label=source_label)
    required = tuple(required_case_keys)

    if df.empty:
        report.missing_canonical = [k for k in required if k not in df.columns]
        return df.copy(), report

    lookup = _header_lookup(df.columns)
    rename_map: Dict[str, str] = {}
    targets_seen: Dict[str, str] = {}

    all_aliases: Dict[str, Tuple[str, ...]] = {k: tuple(v) for k, v in CASE_KEY_ALIASES.items()}
    if extra_aliases:
        for key, aliases in extra_aliases.items():
            base = all_aliases.get(key, ())
            all_aliases[key] = base + tuple(a for a in aliases if a not in base)
    all_aliases["typus"] = tuple(typus_aliases)

    for canonical, aliases in all_aliases.items():
        original = _resolve_alias(lookup, aliases)
        if original is None:
            if canonical in required or canonical == "typus":
                if canonical not in report.missing_canonical:
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

    if "opber_fallnr" in out.columns:
        out["opber_fallnr"] = normalize_excel_pid_series(out["opber_fallnr"], source_label=f"{source_label}.opber_fallnr")

    if normalize_merge_keys:
        out = apply_canonical_merge_key_normalization(out, source_label=source_label)

    if "typus" in out.columns:
        out["typus"] = out["typus"].map(_cell_as_display_string)

    return out, report


def _cell_as_display_string(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).strip()
