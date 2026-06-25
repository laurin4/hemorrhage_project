"""Tests for merging hemorrhage case classifications into the patient/case sheet."""

from __future__ import annotations

import pandas as pd

from src.tasks.hemorrhage.export.classification_merge import (
    CLASS_HAEMORRHAGIC_ACUTE,
    CLASS_HAEMORRHAGIC_HISTORICAL,
    CLASS_HAEMORRHAGIC_NON_ACUTE,
    CLASS_NON_HAEMORRHAGIC,
    CLASS_COLUMNS,
    STATUS_COLUMN,
    STATUS_NOT_IN_PREDICTIONS,
    STATUS_OK,
    STATUS_SUBTYPE_UNKNOWN,
    build_case_classification_map,
    classify_prediction,
    merge_classifications_into_template,
    run_merge_classifications,
)


# --------------------------------------------------------------------------- #
# classify_prediction
# --------------------------------------------------------------------------- #
def test_classify_acute():
    c = classify_prediction("hämorrhagisch", "akut", "success")
    assert c.class_column == CLASS_HAEMORRHAGIC_ACUTE
    assert c.status == STATUS_OK


def test_classify_non_acute():
    c = classify_prediction("hämorrhagisch", "nicht_akut", "success")
    assert c.class_column == CLASS_HAEMORRHAGIC_NON_ACUTE


def test_classify_historical():
    c = classify_prediction("hämorrhagisch", "historisch", "success")
    assert c.class_column == CLASS_HAEMORRHAGIC_HISTORICAL


def test_classify_non_hemorrhagic():
    c = classify_prediction("nicht_hämorrhagisch", "", "success")
    assert c.class_column == CLASS_NON_HAEMORRHAGIC


def test_classify_hemorrhagic_unknown_subtype_has_no_marker():
    c = classify_prediction("hämorrhagisch", "unbekannt", "success")
    assert c.class_column is None
    assert c.status == STATUS_SUBTYPE_UNKNOWN


def test_classify_failed_status_has_no_marker():
    for status in ("parse_failed", "llm_failed", "dry_run"):
        c = classify_prediction("hämorrhagisch", "akut", status)
        assert c.class_column is None
        assert c.status == status


# --------------------------------------------------------------------------- #
# build_case_classification_map
# --------------------------------------------------------------------------- #
def _preds_df(rows):
    cols = ["excel_pid", "excel_opdat", "opber_fallnr", "label", "haemorrhage_subtype", "status"]
    return pd.DataFrame(rows, columns=cols).astype(str)


def test_build_case_map_keys_and_values():
    preds = _preds_df(
        [
            ["1", "2014-05-03", "F1", "hämorrhagisch", "akut", "success"],
            ["2", "2015-01-01", "F2", "nicht_hämorrhagisch", "", "success"],
        ]
    )
    mapping = build_case_classification_map(preds)
    assert mapping[("1", "2014-05-03", "F1")].class_column == CLASS_HAEMORRHAGIC_ACUTE
    assert mapping[("2", "2015-01-01", "F2")].class_column == CLASS_NON_HAEMORRHAGIC


# --------------------------------------------------------------------------- #
# merge_classifications_into_template
# --------------------------------------------------------------------------- #
def _template_df(rows):
    cols = [
        "excel_pid",
        "excel_opdat",
        "opber_fallnr",
        "typus",
        CLASS_HAEMORRHAGIC_ACUTE,
        CLASS_HAEMORRHAGIC_NON_ACUTE,
        CLASS_HAEMORRHAGIC_HISTORICAL,
        CLASS_NON_HAEMORRHAGIC,
    ]
    return pd.DataFrame(rows, columns=cols)


def test_merge_one_hot_and_broadcast_to_report_rows():
    # One case (pid=1, opdat, fallnr=F1) with three report rows (typus 01/02/03).
    template = _template_df(
        [
            ["1", "2014-05-03", "F1", "01", "", "", "", ""],
            ["1", "2014-05-03", "F1", "02", "", "", "", ""],
            ["1", "2014-05-03", "F1", "03", "", "", "", ""],
        ]
    )
    preds = _preds_df([["1", "2014-05-03", "F1", "hämorrhagisch", "akut", "success"]])
    class_map = build_case_classification_map(preds)

    merged, result = merge_classifications_into_template(template, class_map)

    # All three report rows get the same one-hot encoding.
    assert (merged[CLASS_HAEMORRHAGIC_ACUTE] == 1).all()
    for col in (
        CLASS_HAEMORRHAGIC_NON_ACUTE,
        CLASS_HAEMORRHAGIC_HISTORICAL,
        CLASS_NON_HAEMORRHAGIC,
    ):
        assert (merged[col] == 0).all()
    assert (merged[STATUS_COLUMN] == STATUS_OK).all()

    assert result.template_rows == 3
    assert result.classified_rows == 3
    assert result.matched_cases == 1
    assert result.class_row_counts[CLASS_HAEMORRHAGIC_ACUTE] == 3


def test_merge_proper_one_hot_single_one():
    template = _template_df([["7", "2020-02-02", "F7", "01", "", "", "", ""]])
    preds = _preds_df([["7", "2020-02-02", "F7", "hämorrhagisch", "historisch", "success"]])
    merged, _ = merge_classifications_into_template(
        template, build_case_classification_map(preds)
    )
    row = merged.iloc[0]
    one_hot = [row[c] for c in CLASS_COLUMNS]
    assert sum(int(v) for v in one_hot) == 1
    assert row[CLASS_HAEMORRHAGIC_HISTORICAL] == 1


def test_merge_unmatched_row_left_blank_with_status():
    template = _template_df([["99", "1999-09-09", "F99", "01", "", "", "", ""]])
    preds = _preds_df([["1", "2014-05-03", "F1", "hämorrhagisch", "akut", "success"]])
    merged, result = merge_classifications_into_template(
        template, build_case_classification_map(preds)
    )
    row = merged.iloc[0]
    for col in CLASS_COLUMNS:
        assert row[col] == ""
    assert row[STATUS_COLUMN] == STATUS_NOT_IN_PREDICTIONS
    assert result.unmatched_rows == 1
    assert result.classified_rows == 0


def test_merge_failed_case_blank_with_failure_status():
    template = _template_df([["5", "2018-08-08", "F5", "01", "", "", "", ""]])
    preds = _preds_df([["5", "2018-08-08", "F5", "", "", "parse_failed"]])
    merged, result = merge_classifications_into_template(
        template, build_case_classification_map(preds)
    )
    row = merged.iloc[0]
    for col in CLASS_COLUMNS:
        assert row[col] == ""
    assert row[STATUS_COLUMN] == "parse_failed"
    assert result.unclassified_rows == 1
    assert result.matched_rows == 1


def test_merge_key_normalization_pid_float_and_iso_date():
    # Template stores pid as float and opdat as a Timestamp; predictions store
    # the normalized string forms. They must still match.
    template = pd.DataFrame(
        [[1.0, pd.Timestamp("2014-05-03"), "F1", "01", "", "", "", ""]],
        columns=[
            "excel_pid",
            "excel_opdat",
            "opber_fallnr",
            "typus",
            CLASS_HAEMORRHAGIC_ACUTE,
            CLASS_HAEMORRHAGIC_NON_ACUTE,
            CLASS_HAEMORRHAGIC_HISTORICAL,
            CLASS_NON_HAEMORRHAGIC,
        ],
    )
    preds = _preds_df([["1", "2014-05-03", "F1", "hämorrhagisch", "akut", "success"]])
    merged, result = merge_classifications_into_template(
        template, build_case_classification_map(preds)
    )
    assert result.classified_rows == 1
    assert merged.iloc[0][CLASS_HAEMORRHAGIC_ACUTE] == 1


# --------------------------------------------------------------------------- #
# run_merge_classifications (end-to-end with temp files)
# --------------------------------------------------------------------------- #
def test_run_merge_end_to_end(tmp_path):
    pred_path = tmp_path / "preds.csv"
    _preds_df(
        [
            ["1", "2014-05-03", "F1", "hämorrhagisch", "nicht_akut", "success"],
            ["2", "2015-01-01", "F2", "nicht_hämorrhagisch", "", "success"],
        ]
    ).to_csv(pred_path, index=False)

    tmpl_path = tmp_path / "template.xlsx"
    _template_df(
        [
            ["1", "2014-05-03", "F1", "01", "", "", "", ""],
            ["1", "2014-05-03", "F1", "02", "", "", "", ""],
            ["2", "2015-01-01", "F2", "01", "", "", "", ""],
            ["3", "2016-06-06", "F3", "01", "", "", "", ""],  # no prediction
        ]
    ).to_excel(tmpl_path, index=False, engine="openpyxl")

    out_path = tmp_path / "merged.xlsx"
    result = run_merge_classifications(
        predictions_path=pred_path,
        template_path=tmpl_path,
        output_path=out_path,
        summary_path=tmp_path / "summary.txt",
        unmatched_path=tmp_path / "unmatched.csv",
    )

    assert out_path.exists()
    assert result.template_rows == 4
    assert result.classified_rows == 3
    assert result.unmatched_rows == 1
    assert result.matched_cases == 2

    merged = pd.read_excel(out_path, engine="openpyxl")
    case1 = merged[merged["opber_fallnr"] == "F1"]
    assert (case1[CLASS_HAEMORRHAGIC_NON_ACUTE] == 1).all()
    case3 = merged[merged["opber_fallnr"] == "F3"]
    assert (case3[STATUS_COLUMN] == STATUS_NOT_IN_PREDICTIONS).all()

    # Column order: record columns first, then one-hot classes, then status last.
    cols = list(merged.columns)
    assert cols[-1] == STATUS_COLUMN
    assert cols[-5:-1] == list(CLASS_COLUMNS)
    # 'typus' (a record column) must come before the class block.
    assert cols.index("typus") < cols.index(CLASS_HAEMORRHAGIC_ACUTE)


def test_merge_reorders_class_columns_to_right():
    # Template with class columns deliberately placed in the MIDDLE.
    template = pd.DataFrame(
        [["1", "2014-05-03", "F1", "01", "", "", "", "", "Mueller", "diag text"]],
        columns=[
            "excel_pid",
            "excel_opdat",
            "opber_fallnr",
            "typus",
            CLASS_HAEMORRHAGIC_ACUTE,
            CLASS_HAEMORRHAGIC_NON_ACUTE,
            CLASS_HAEMORRHAGIC_HISTORICAL,
            CLASS_NON_HAEMORRHAGIC,
            "bername",
            "diag",
        ],
    )
    preds = _preds_df([["1", "2014-05-03", "F1", "hämorrhagisch", "akut", "success"]])
    merged, _ = merge_classifications_into_template(
        template, build_case_classification_map(preds)
    )
    cols = list(merged.columns)
    # Record columns (incl. bername/diag) come before the class block.
    assert cols.index("bername") < cols.index(CLASS_HAEMORRHAGIC_ACUTE)
    assert cols.index("diag") < cols.index(CLASS_HAEMORRHAGIC_ACUTE)
    assert cols[-5:] == list(CLASS_COLUMNS) + [STATUS_COLUMN]
