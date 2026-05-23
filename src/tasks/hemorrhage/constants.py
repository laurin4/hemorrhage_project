"""
Hemorrhage task constants — case keys, report typus codes, column names.

No NLP / keyword / prompt content in this module.
"""

from __future__ import annotations

from typing import Dict, Tuple

# Case grouping keys (clinical case definition)
CASE_KEY_COLUMNS: Tuple[str, str, str] = ("excel_pid", "excel_opdat", "opber_fallnr")

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

# Row identity within flat file (optional)
SOURCE_ROW_ID_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "source_report_row_id",
    "row_id",
    "berichte_row_id",
)
