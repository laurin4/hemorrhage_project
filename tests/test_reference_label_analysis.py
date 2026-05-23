"""Tests for reference label analytics."""

from pathlib import Path

import pandas as pd

from src.tasks.hemorrhage.analysis.reference_labels import (
    assign_reference_label_stratum,
    parse_label_value,
    resolve_label_columns,
    run_reference_label_analysis,
)


def test_parse_label_yes_variants():
    assert parse_label_value("Ja")[0] == "yes"
    assert parse_label_value("JA")[0] == "yes"
    assert parse_label_value(1)[0] == "yes"
    assert parse_label_value("yes")[0] == "yes"
    assert parse_label_value(True)[0] == "yes"


def test_parse_label_missing_and_nonstandard():
    assert parse_label_value("")[0] == "missing"
    assert parse_label_value(None)[0] == "missing"
    assert parse_label_value("maybe")[0] == "non_standard"


def test_stratum_assignment():
    assert assign_reference_label_stratum("yes", "no") == "hemorrhagic"
    assert assign_reference_label_stratum("no", "yes") == "non_hemorrhagic"
    assert assign_reference_label_stratum("yes", "yes") == "both_marked"
    assert assign_reference_label_stratum("missing", "missing") == "unlabeled"


def test_label_balance_and_inconsistencies(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "Patient::Patientennummer": "1",
                "v_Operation_Datum": "2024-01-01",
                "Hämorrhagisch": "Ja",
                "Nicht Hämorrhagisch": "",
                "Verify_Vaskulär": 1,
                "Indikation 1": "CCM Blutung",
                "Eingriff": "OP",
            },
            {
                "Patient::Patientennummer": "2",
                "v_Operation_Datum": "2024-01-02",
                "Hämorrhagisch": "Ja",
                "Nicht Hämorrhagisch": "Ja",
                "Verify_Vaskulär": "",
                "Indikation 1": "DAVF",
                "Eingriff": "Embolisation",
            },
            {
                "Patient::Patientennummer": "3",
                "v_Operation_Datum": "2024-01-03",
                "Hämorrhagisch": "",
                "Nicht Hämorrhagisch": "",
                "Verify_Vaskulär": "nein",
                "Indikation 1": "unsicher",
                "Eingriff": "",
            },
            {
                "Patient::Patientennummer": "4",
                "v_Operation_Datum": "2024-01-04",
                "Hämorrhagisch": "maybe",
                "Nicht Hämorrhagisch": "Nein",
                "Verify_Vaskulär": "Ja",
                "Indikation 1": "Cavernom",
                "Eingriff": "Resektion",
            },
        ]
    )
    xlsx = tmp_path / "260507_CCM_DAVF.xlsx"
    df.to_excel(xlsx, index=False)

    out_dir = tmp_path / "inspection"
    result = run_reference_label_analysis(reference_path=xlsx, output_dir=out_dir)

    assert result.total_rows == 4
    assert (out_dir / "reference_label_summary.csv").exists()
    assert (out_dir / "reference_label_inconsistencies.csv").exists()

    balance = pd.read_csv(out_dir / "reference_label_summary.csv")
    hemo_yes = int(balance.loc[balance["metric"] == "haemorrhagisch_yes", "count"].iloc[0])
    both = int(balance.loc[balance["metric"] == "both_haemorrhagisch_and_nicht_yes", "count"].iloc[0])
    assert hemo_yes == 2
    assert both == 1

    inc = pd.read_csv(out_dir / "reference_label_inconsistencies.csv")
    assert len(inc) >= 3

    kw = pd.read_csv(out_dir / "reference_keyword_by_label.csv")
    ccm_rows = kw[(kw["keyword"] == "ccm") & (kw["rows_with_keyword"] > 0)]
    assert not ccm_rows.empty

    label_cols = resolve_label_columns(df)
    assert label_cols["haemorrhagisch"] == "Hämorrhagisch"
