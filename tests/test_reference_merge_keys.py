"""Reference column aliases and (excel_pid, excel_opdat) merge linkage."""

import pandas as pd

from src.tasks.hemorrhage.constants import REFERENCE_KEY_ALIASES_EXTRA
from src.tasks.hemorrhage.inspection.merge_validation import merge_validation
from src.tasks.hemorrhage.io.column_normalize import normalize_dataframe_columns
from src.tasks.hemorrhage.io.key_normalize import merge_reference_key_aliases
from src.tasks.hemorrhage.constants import CASE_KEY_ALIASES
from src.tasks.hemorrhage.preprocessing.case_builder import build_cases_from_dataframe


def test_reference_ccm_column_aliases():
    df = pd.DataFrame(
        [
            {
                "Patient::Patientennummer": "100",
                "v_Operation_Datum": pd.Timestamp("2024-01-10"),
                "Hämorrhagisch": 1,
            }
        ]
    )
    aliases = merge_reference_key_aliases(CASE_KEY_ALIASES, REFERENCE_KEY_ALIASES_EXTRA)
    out, report = normalize_dataframe_columns(
        df,
        source_label="reference",
        extra_aliases=aliases,
        required_case_keys=("excel_pid", "excel_opdat"),
    )
    assert "excel_pid" in out.columns
    assert "excel_opdat" in out.columns
    assert out.iloc[0]["excel_pid"] == "100"
    assert out.iloc[0]["excel_opdat"] == "2024-01-10"
    mapped = {m["canonical_column"]: m for m in report.mappings if m["status"] == "mapped"}
    assert mapped["excel_pid"]["original_column"] == "Patient::Patientennummer"
    assert mapped["excel_opdat"]["original_column"] == "v_Operation_Datum"
    assert "excel_pid" not in report.missing_canonical
    assert "excel_opdat" not in report.missing_canonical


def test_merge_on_pid_opdat_despite_missing_reference_fallnr():
    reports = pd.DataFrame(
        [
            {
                "excel_pid": "100",
                "excel_opdat": "2024-01-10",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "diag": "x",
            }
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "Patient::Patientennummer": 100,
                "v_Operation_Datum": "2024-01-10",
                "Hämorrhagisch": 1,
            }
        ]
    )
    rep_out, _ = normalize_dataframe_columns(reports, source_label="reports")
    ref_aliases = merge_reference_key_aliases(CASE_KEY_ALIASES, REFERENCE_KEY_ALIASES_EXTRA)
    ref_out, _ = normalize_dataframe_columns(
        reference,
        source_label="reference",
        extra_aliases=ref_aliases,
        required_case_keys=("excel_pid", "excel_opdat"),
    )
    cases, _ = build_cases_from_dataframe(rep_out)
    summary, un_ref, un_rep, _ = merge_validation(ref_out, rep_out, cases)
    matched = int(summary.loc[summary["metric"] == "matched_link_keys", "value"].iloc[0])
    assert matched == 1
    assert len(un_ref) == 0
    assert len(un_rep) == 0


def test_opdat_datetime_reports_matches_reference_string():
    reports = pd.DataFrame(
        [
            {
                "excel_pid": "200",
                "excel_opdat": pd.Timestamp("2024-02-02"),
                "opber_fallnr": "F2",
                "typus": "02 Eintrittsbericht",
                "diag": "a",
            }
        ]
    )
    reference = pd.DataFrame(
        [{"Patient::Patientennummer": "200", "v_Operation_Datum": "02.02.2024"}]
    )
    rep_out, _ = normalize_dataframe_columns(reports, source_label="reports")
    ref_aliases = merge_reference_key_aliases(CASE_KEY_ALIASES, REFERENCE_KEY_ALIASES_EXTRA)
    ref_out, _ = normalize_dataframe_columns(
        reference,
        source_label="reference",
        extra_aliases=ref_aliases,
        required_case_keys=("excel_pid", "excel_opdat"),
    )
    assert rep_out.iloc[0]["excel_opdat"] == "2024-02-02"
    assert ref_out.iloc[0]["excel_opdat"] == "2024-02-02"
    cases, _ = build_cases_from_dataframe(rep_out)
    summary, _, _, _ = merge_validation(ref_out, rep_out, cases)
    matched = int(summary.loc[summary["metric"] == "matched_link_keys", "value"].iloc[0])
    assert matched == 1
