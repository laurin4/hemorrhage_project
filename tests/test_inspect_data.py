"""Inspection pipeline tests with synthetic Excel inputs."""

from pathlib import Path

import pandas as pd
import pytest

from src.tasks.hemorrhage.inspection.runner import run_full_inspection
from src.tasks.hemorrhage.io.column_normalize import normalize_dataframe_columns
from src.tasks.hemorrhage.preprocessing.case_builder import build_cases_from_dataframe


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> tuple[Path, Path]:
    reports = pd.DataFrame(
        [
            {
                "excel_pid": "100",
                "excel_opdat": "2024-01-10",
                "opber_fallnr": "F1",
                "typus": "01 Operationsbericht",
                "diag": "CCM Blutung hämorrhagisch",
                "indik_untersuch": "",
                "vorgehen_beurt": "OP durchgeführt",
            },
            {
                "excel_pid": "100",
                "excel_opdat": "2024-01-10",
                "opber_fallnr": "F1",
                "typus": "03 Austrittsbericht",
                "diag": "Verlauf stabil",
                "indik_untersuch": "",
                "vorgehen_beurt": "",
            },
            {
                "excel_pid": "200",
                "excel_opdat": "2024-02-02",
                "opber_fallnr": "F2",
                "typus": "02 Eintrittsbericht",
                "diag": "DAVF Verdacht",
                "indik_untersuch": "MRI",
                "vorgehen_beurt": "",
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "Patient::Patientennummer": "100",
                "v_Operation_Datum": "2024-01-10",
                "Hämorrhagisch": 1,
                "Nicht Hämorrhagisch": 0,
            },
            {
                "Patient::Patientennummer": "999",
                "v_Operation_Datum": "2024-03-03",
                "Hämorrhagisch": 0,
            },
        ]
    )
    rep_path = tmp_path / "NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx"
    ref_path = tmp_path / "260507_CCM_DAVF.xlsx"
    reports.to_excel(rep_path, index=False)
    reference.to_excel(ref_path, index=False)
    return rep_path, ref_path


def test_normalize_maps_aliases():
    df = pd.DataFrame([{"PatientID": "1", "op_datum": "2024-01-01", "fall_nr": "F", "Typus": "01 OP"}])
    out, rep = normalize_dataframe_columns(df, source_label="test")
    assert "excel_pid" in out.columns
    assert rep.mappings


def test_inspection_pipeline(synthetic_raw_dir, tmp_path: Path):
    rep_path, ref_path = synthetic_raw_dir
    out_dir = tmp_path / "inspection"
    result = run_full_inspection(
        reports_path=rep_path,
        reference_path=ref_path,
        output_dir=out_dir,
    )
    assert result.cases_built == 2
    assert result.incomplete_cases == 2  # OP+Austritt only; Eintritt only
    assert result.complete_cases == 0
    assert (out_dir / "case_summary.csv").exists()
    assert (out_dir / "merge_validation.csv").exists()
    assert (out_dir / "keyword_exploration.csv").exists()
    assert (out_dir / "structured_case_samples.csv").exists()
    merge = pd.read_csv(out_dir / "merge_validation.csv")
    matched = int(merge.loc[merge["metric"] == "matched_link_keys", "value"].iloc[0])
    assert matched >= 1
    assert (out_dir / "unmatched_reference_rows.csv").exists()


def test_incomplete_case_pattern():
    df = pd.DataFrame(
        [
            {
                "excel_pid": "1",
                "excel_opdat": "2024-01-01",
                "opber_fallnr": "F",
                "typus": "01 Operationsbericht",
                "diag": "x",
            }
        ]
    )
    cases, _ = build_cases_from_dataframe(df)
    assert len(cases) == 1
    assert cases[0].missing_report_types
