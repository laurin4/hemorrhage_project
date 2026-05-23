"""
Hemorrhage task constants — case keys, report typus codes, column names.

No NLP / keyword / prompt content in this module.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Case grouping keys (clinical case definition)
CASE_KEY_COLUMNS: Tuple[str, str, str] = ("excel_pid", "excel_opdat", "opber_fallnr")

# Reference ↔ reports linkage (CCM DAVF sheet has no opber_fallnr)
MERGE_LINK_KEY_COLUMNS: Tuple[str, str] = ("excel_pid", "excel_opdat")

# Canonical report typus codes (prefix "01", "02", "03")
TYPUS_OPERATIONSBERICHT = "01"
TYPUS_EINTRITTSBERICHT = "02"
TYPUS_AUSTRITTSBERICHT = "03"

EXPECTED_REPORT_TYPUS_CODES: Tuple[str, ...] = (
    TYPUS_OPERATIONSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_AUSTRITTSBERICHT,
)

TYPUS_CODE_TO_LABEL: Dict[str, str] = {
    TYPUS_OPERATIONSBERICHT: "01 Operationsbericht",
    TYPUS_EINTRITTSBERICHT: "02 Eintrittsbericht",
    TYPUS_AUSTRITTSBERICHT: "03 Austrittsbericht",
}

# Raw ``typus`` column aliases (first match wins)
TYPUS_COLUMN_CANDIDATES: Tuple[str, ...] = ("typus", "Typus", "bertyp", "report_typus")

# CCM DAVF clinical export — primary text fields for inspection / future NLP
HEMORRHAGE_TEXT_FIELDS: Tuple[str, ...] = ("diag", "indik_untersuch", "vorgehen_beurt")

# Optional text columns for report body (stitched in order when multiple present)
TEXT_COLUMN_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("diag", "[Diagnosen]"),
    ("indik_untersuch", "[Indikation/Untersuch]"),
    ("vorgehen_beurt", "[Vorgehen/Beurteilung]"),
    ("report_text", "[Volltext]"),
    ("epikrise", "[Epikrise]"),
    ("jetziges_leiden", "[Jetziges Leiden]"),
    ("prozedere", "[Prozedere]"),
    ("befund", "[Befund]"),
    ("verlauf", "[Verlauf]"),
)

# Column alias → canonical name (case-insensitive match on Excel headers)
CASE_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "excel_pid": ("excel_pid", "excel-pid", "pid", "patientid", "patient_id", "patientenid"),
    "excel_opdat": (
        "excel_opdat",
        "excel-opdat",
        "opdat",
        "op_datum",
        "operation_date",
        "op_date",
    ),
    "opber_fallnr": (
        "opber_fallnr",
        "opber-fallnr",
        "fallnr",
        "fall_nr",
        "opbericht_fallnr",
        "op_fallnr",
    ),
}

# CCM DAVF reference sheet (260507) — merged with CASE_KEY_ALIASES when normalizing reference
REFERENCE_KEY_ALIASES_EXTRA: Dict[str, Tuple[str, ...]] = {
    "excel_pid": ("patient::patientennummer",),
    "excel_opdat": ("v_operation_datum",),
}

REFERENCE_REQUIRED_CANONICAL_KEYS: Tuple[str, ...] = MERGE_LINK_KEY_COLUMNS

TYPUS_ALIASES: Tuple[str, ...] = (
    "typus",
    "Typus",
    "report_typus",
    "berichtstyp",
    "bertyp",
    "dokumenttyp",
)

# Exploratory keywords (counts only — no classification)
KEYWORD_EXPLORATION_TERMS: Tuple[str, ...] = (
    "hämorrhag",
    "hemorrhag",
    "blutung",
    "ccm",
    "davf",
    "cavernom",
    "vaskulär",
    "vaskulaer",
)

# Reference label columns (260507 CCM DAVF.xlsx — original header names for lookup)
REFERENCE_LABEL_COLUMN_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "haemorrhagisch": ("hämorrhagisch", "hamorrhagisch", "haemorrhagisch"),
    "nicht_haemorrhagisch": ("nicht hämorrhagisch", "nicht hamorrhagisch", "nicht haemorrhagisch"),
    "verify_vaskulaer": ("verify_vaskulär", "verify_vaskular", "verify vaskulär", "verify vaskular"),
}

REFERENCE_INDICATION_COLUMNS: Tuple[str, ...] = (
    "Indikation 1",
    "Indikation1_Korrigiert",
    "Eingriff",
)

REFERENCE_LABEL_TEXT_COLUMNS: Tuple[str, ...] = REFERENCE_INDICATION_COLUMNS

REFERENCE_KEYWORD_BY_LABEL_TERMS: Tuple[str, ...] = (
    "cavernom",
    "ccm",
    "davf",
    "blutung",
    "einblutung",
    "hämorrhag",
    "haemorrhag",
    "hemorrhag",
    "hämosiderin",
    "haemosiderin",
    "vaskulär",
    "vaskulaer",
)

# Spreadsheet yes-values (case-insensitive); descriptive only
REFERENCE_LABEL_YES_VALUES: frozenset[str] = frozenset(
    {"ja", "yes", "1", "true", "y", "x", "wahr"}
)

# Row identity within flat file (optional)
SOURCE_ROW_ID_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "source_report_row_id",
    "row_id",
    "berichte_row_id",
)
