"""Optional reference label fields for manual comparison (not used for prediction)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.tasks.hemorrhage.analysis.reference_labels import (
    assign_reference_label_stratum,
    load_normalized_reference,
    parse_label_value,
    resolve_label_columns,
)
from src.tasks.hemorrhage.config import (
    REFERENCE_XLSX_ALTERNATE_FILENAMES,
    configured_reference_xlsx_path,
)
from src.tasks.hemorrhage.io.path_resolve import resolve_raw_input_path

ReferenceLookup = Dict[Tuple[str, str], Dict[str, str]]


def resolve_reference_path(reference_path: Path | None = None) -> Path:
    """Return resolved reference Excel path (may be missing on disk)."""
    configured = reference_path or configured_reference_xlsx_path()
    resolved = resolve_raw_input_path(
        configured, REFERENCE_XLSX_ALTERNATE_FILENAMES, context="reference"
    )
    return resolved.resolved_path


def _cell_raw(row: pd.Series, col: Optional[str]) -> str:
    if not col:
        return ""
    val = row.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def build_reference_lookup(
    reference_path=None,
) -> Tuple[ReferenceLookup, List[str], Path]:
    """
    Map (excel_pid, excel_opdat) → reference label display fields.

    Returns empty dict if reference unavailable — pipeline continues.
    """
    errors: List[str] = []
    resolved_path = resolve_reference_path(reference_path)
    df, _, load_errors = load_normalized_reference(reference_path)
    errors.extend(load_errors)
    if df.empty:
        return {}, errors, resolved_path

    label_cols = resolve_label_columns(df)
    lookup: ReferenceLookup = {}

    for _, row in df.iterrows():
        pid = str(row.get("excel_pid", "") or "").strip()
        opdat = str(row.get("excel_opdat", "") or "").strip()
        key = (pid, opdat)

        hemo_col = label_cols.get("haemorrhagisch")
        non_col = label_cols.get("nicht_haemorrhagisch")
        ver_col = label_cols.get("verify_vaskulaer")

        h_state, _ = parse_label_value(row.get(hemo_col)) if hemo_col else ("missing", "")
        n_state, _ = parse_label_value(row.get(non_col)) if non_col else ("missing", "")

        lookup[key] = {
            "reference_haemorrhagisch": _cell_raw(row, hemo_col),
            "reference_nicht_haemorrhagisch": _cell_raw(row, non_col),
            "reference_verify_vaskulaer": _cell_raw(row, ver_col),
            "reference_label_status": assign_reference_label_stratum(h_state, n_state),
            # No validated reference subtype labels exist yet; kept empty as metadata
            # placeholder until a future reference column is added.
            "reference_haemorrhage_subtype": "",
        }

    return lookup, errors, resolved_path


# Reference statuses that constitute the binary-labeled evaluation cohort.
LABELED_BINARY_STATUSES = frozenset({"hemorrhagic", "non_hemorrhagic"})


def reference_binary_status(ref_fields: Dict[str, str]) -> str:
    """
    Canonical reference status used for cohort filtering and evaluation.

    Derived from the three raw spreadsheet label cells:
    - Hämorrhagisch only → ``hemorrhagic``
    - Nicht Hämorrhagisch only → ``non_hemorrhagic``
    - only Verify_Vaskulär → ``verify_only``
    - both Hämo + Nicht Hämo → ``inconsistent``
    - everything else (incl. reference-not-found) → ``unknown``
    """
    h_state, _ = parse_label_value(ref_fields.get("reference_haemorrhagisch"))
    n_state, _ = parse_label_value(ref_fields.get("reference_nicht_haemorrhagisch"))
    v_state, _ = parse_label_value(ref_fields.get("reference_verify_vaskulaer"))

    if h_state == "yes" and n_state == "yes":
        return "inconsistent"
    if h_state == "yes" and n_state != "yes":
        return "hemorrhagic"
    if n_state == "yes" and h_state != "yes":
        return "non_hemorrhagic"
    if v_state == "yes" and h_state != "yes" and n_state != "yes":
        return "verify_only"
    return "unknown"


def reference_fields_for_case(
    lookup: ReferenceLookup,
    excel_pid: str,
    excel_opdat: str,
) -> Dict[str, str]:
    key = (str(excel_pid or "").strip(), str(excel_opdat or "").strip())
    return lookup.get(
        key,
        {
            "reference_haemorrhagisch": "",
            "reference_nicht_haemorrhagisch": "",
            "reference_verify_vaskulaer": "",
            "reference_label_status": "reference_not_found",
            "reference_haemorrhage_subtype": "",
        },
    )
