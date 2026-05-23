"""
Canonical merge-key normalization (excel_pid, excel_opdat).

Shared by reports and reference so linkage compares stable string forms.
Invalid dates are not silently coerced to empty — they are logged and kept as stripped raw strings.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

ISO_DATE_FMT = "%Y-%m-%d"


def merge_reference_key_aliases(
    base: Dict[str, Tuple[str, ...]],
    extra: Dict[str, Tuple[str, ...]],
) -> Dict[str, Tuple[str, ...]]:
    """Combine base + extra alias tuples per canonical column."""
    merged: Dict[str, Tuple[str, ...]] = {k: tuple(v) for k, v in base.items()}
    for key, aliases in extra.items():
        existing = merged.get(key, ())
        merged[key] = existing + tuple(a for a in aliases if a not in existing)
    return merged


def normalize_excel_pid_series(series: pd.Series, *, source_label: str) -> pd.Series:
    """Strip and stringify patient ids; log non-string dtypes."""
    if series.empty:
        return series.astype(str)

    non_null = series.dropna()
    if len(non_null) and not pd.api.types.is_string_dtype(series):
        LOGGER.info(
            "[%s] excel_pid dtype=%s — normalizing to string (sample=%r).",
            source_label,
            series.dtype,
            non_null.iloc[0],
        )

    def _one(v: object) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v).strip()

    return series.map(_one)


def normalize_excel_opdat_series(series: pd.Series, *, source_label: str) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Normalize operation dates to ``YYYY-MM-DD`` strings when parseable.

    Returns (normalized_series, stats_dict) for logging.
    """
    stats = {
        "empty": 0,
        "from_datetime_dtype": 0,
        "parsed_ok": 0,
        "parse_failed_kept_raw": 0,
        "already_iso_like": 0,
    }
    if series.empty:
        return series.astype(str), stats

    out: list[str] = []
    for v in series:
        norm, kind = _normalize_opdat_cell(v)
        stats[kind] = stats.get(kind, 0) + 1
        out.append(norm)

    if stats["from_datetime_dtype"] or stats["parse_failed_kept_raw"]:
        LOGGER.info(
            "[%s] excel_opdat normalization: datetime_dtype=%d parsed_ok=%d "
            "parse_failed_kept_raw=%d already_iso_like=%d empty=%d",
            source_label,
            stats.get("from_datetime_dtype", 0),
            stats.get("parsed_ok", 0),
            stats.get("parse_failed_kept_raw", 0),
            stats.get("already_iso_like", 0),
            stats.get("empty", 0),
        )

    return pd.Series(out, index=series.index), stats


def _normalize_opdat_cell(value: object) -> Tuple[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "", "empty"

    if isinstance(value, pd.Timestamp):
        return value.strftime(ISO_DATE_FMT), "from_datetime_dtype"

    if hasattr(value, "strftime") and not isinstance(value, str):
        try:
            return value.strftime(ISO_DATE_FMT), "from_datetime_dtype"
        except Exception:
            pass

    if isinstance(value, float):
        if pd.isna(value):
            return "", "empty"
        # Excel serial date heuristic (reasonable range)
        if 30000 < value < 60000:
            try:
                ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
                return ts.strftime(ISO_DATE_FMT), "parsed_ok"
            except Exception:
                pass
        if value == int(value):
            return str(int(value)), "already_iso_like"

    raw = str(value).strip()
    if not raw or raw.lower() in ("nan", "none", "<na>"):
        return "", "empty"

    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10], "already_iso_like"

    parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return parsed.strftime(ISO_DATE_FMT), "parsed_ok"

    return raw, "parse_failed_kept_raw"


def apply_canonical_merge_key_normalization(
    df: pd.DataFrame,
    *,
    source_label: str,
) -> pd.DataFrame:
    """Apply ``excel_pid`` / ``excel_opdat`` normalization in place on a copy."""
    out = df.copy()
    if "excel_pid" in out.columns:
        out["excel_pid"] = normalize_excel_pid_series(out["excel_pid"], source_label=source_label)
    if "excel_opdat" in out.columns:
        out["excel_opdat"], _ = normalize_excel_opdat_series(
            out["excel_opdat"], source_label=source_label
        )
    return out
