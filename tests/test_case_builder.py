"""Tests for case-centric construction (incomplete cases, duplicates, missing typus)."""

import pandas as pd
import pytest

from src.core.case.keys import MISSING_KEY_TOKEN
from src.tasks.hemorrhage.constants import (
    TYPUS_AUSTRITTSBERICHT,
    TYPUS_EINTRITTSBERICHT,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.export.case_export_schema import (
    CASE_EXPORT_COLUMNS,
    cases_to_export_dataframe,
)
from src.tasks.hemorrhage.inference_policy import prefilter_skip_allowed
from src.tasks.hemorrhage.preprocessing.case_builder import build_cases_from_dataframe


def test_prefilter_disabled_by_default():
    assert prefilter_skip_allowed() is False


def test_single_report_case():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "P1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "report_text": "OP Blutung",
            }
        ]
    )
    cases, stats = build_cases_from_dataframe(df)
    assert len(cases) == 1
    c = cases[0]
    assert c.n_reports_available == 1
    assert TYPUS_OPERATIONSBERICHT in c.available_report_types
    assert TYPUS_EINTRITTSBERICHT in c.missing_report_types
    assert TYPUS_AUSTRITTSBERICHT in c.missing_report_types
    assert not c.is_complete
    assert stats.cases_incomplete == 1


def test_all_three_reports_complete():
    rows = []
    for typus, text in (
        ("01 Operationsbericht", "OP"),
        ("02 Eintrittsbericht", "Eintritt"),
        ("03 Austrittsbericht", "Austritt"),
    ):
        rows.append(
            {
                "excel_pid": "P2",
                "excel_opdat": "2024-02-02",
                "opber_fallnr": "F2",
                "typus": typus,
                "report_text": text,
            }
        )
    cases, stats = build_cases_from_dataframe(pd.DataFrame(rows))
    assert len(cases) == 1
    assert cases[0].is_complete
    assert cases[0].n_reports_available == 3
    assert stats.cases_complete == 1


def test_missing_eintritt_and_austritt():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "P3",
                "excel_opdat": "2024-03-03",
                "opber_fallnr": "F3",
                "typus": "01 Operationsbericht",
                "report_text": "nur OP",
            }
        ]
    )
    cases, _ = build_cases_from_dataframe(df)
    c = cases[0]
    assert c.missing_report_types == (TYPUS_EINTRITTSBERICHT, TYPUS_AUSTRITTSBERICHT)


def test_duplicate_typus_keeps_first():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "P4",
                "excel_opdat": "2024-04-04",
                "opber_fallnr": "F4",
                "typus": "02 Eintrittsbericht",
                "report_text": "first",
            },
            {
                "excel_pid": "P4",
                "excel_opdat": "2024-04-04",
                "opber_fallnr": "F4",
                "typus": "02 Eintrittsbericht",
                "report_text": "second",
            },
        ]
    )
    cases, stats = build_cases_from_dataframe(df)
    assert len(cases) == 1
    assert "first" in cases[0].get_report_text(TYPUS_EINTRITTSBERICHT)
    assert "second" not in cases[0].get_report_text(TYPUS_EINTRITTSBERICHT)
    assert stats.duplicate_typus_in_case == 1


def test_case_with_missing_key_component_preserved():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "",
                "excel_opdat": "2024-05-05",
                "opber_fallnr": "F5",
                "typus": "03 Austrittsbericht",
                "report_text": "Austritt only",
            }
        ]
    )
    cases, stats = build_cases_from_dataframe(df)
    assert len(cases) == 1
    assert cases[0].excel_pid == MISSING_KEY_TOKEN
    assert stats.rows_with_missing_key_component == 1


def test_export_one_row_per_case():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "PA",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "1",
                "typus": "01 Operationsbericht",
                "report_text": "a",
            },
            {
                "excel_pid": "PB",
                "excel_opdat": "2024-01-02",
                "opber_fallnr": "2",
                "typus": "02 Eintrittsbericht",
                "report_text": "b",
            },
        ]
    )
    cases, _ = build_cases_from_dataframe(df)
    export_df = cases_to_export_dataframe(cases)
    assert len(export_df) == 2
    assert list(export_df.columns) == CASE_EXPORT_COLUMNS
    assert export_df.iloc[0]["prediction_status"] == "not_run"
