"""Tests for the interactive extraction demo (rendering + selection, no live LLM)."""

from __future__ import annotations

import json

import pandas as pd

from src.core.case.keys import CaseKey
from src.core.case.models import CaseReport, ClinicalCase
from src.tasks.hemorrhage.demo import (
    _autopick_negative_from_predictions,
    _collapse_evidence,
    _evidenz_from_stage,
    _render_cited_passages,
    _select_polarity_case,
    _split_report_sections,
    load_trace,
    present_case,
)
from src.tasks.hemorrhage.demo_extraction import build_trace
from src.tasks.hemorrhage.export.classification_merge import (
    CLASS_HAEMORRHAGIC_NON_ACUTE,
    CLASS_NON_HAEMORRHAGIC,
)


def _make_case(case_id: str, text: str) -> ClinicalCase:
    return ClinicalCase(
        case_key=CaseKey(case_id, "2014-05-03", "F1"),
        case_id=case_id,
        reports={
            "01": CaseReport(typus_code="01", typus_label="01 Operationsbericht", report_text=text),
        },
        available_report_types=("01",),
    )


def _positive_trace() -> dict:
    case = _make_case(
        "pos_case",
        "[Diagnosen]\nEingeblutetes Kavernom, führte zur Resektion.\n\n[Vorgehen/Beurteilung]\nHämatom nachweisbar.",
    )
    return build_trace(
        case,
        binary_raw='{"klasse": 1, "label": "hämorrhagisch", "sicherheit": "hoch", "kurzbegruendung": "Einblutung"}',
        subtype_raw=(
            '{"haemorrhage_subtype": "nicht_akut", "sicherheit": "hoch", '
            '"begruendung": "Ältere relevante Blutung.", "evidenz": ['
            '{"berichttyp": "Operationsbericht", "feld": "Diagnosen", '
            '"textstelle": "Eingeblutetes Kavernom", '
            '"interpretation": "Prior bleed still relevant to current surgery."}]}'
        ),
        stage2_ran=True,
        mode="TEST",
    )


def _negative_trace() -> dict:
    case = _make_case("neg_case", "Kavernom ohne Einblutung, elektive Resektion bei Epilepsie.")
    return build_trace(
        case,
        binary_raw='{"klasse": 0, "label": "nicht_hämorrhagisch", "sicherheit": "mittel", "kurzbegruendung": "keine Blutung"}',
        subtype_raw="",
        stage2_ran=False,
        mode="TEST",
    )


# --------------------------------------------------------------------------- #
# present_case rendering
# --------------------------------------------------------------------------- #
def test_present_positive_runs_both_stages(capsys):
    present_case(_positive_trace(), pause=False)
    out = capsys.readouterr().out
    assert "STEP 1" in out and "STEP 7" in out
    assert "Stage 2 prompt" in out  # Step 5 title
    assert "STAGE 2 SKIPPED" not in out
    assert CLASS_HAEMORRHAGIC_NON_ACUTE in out
    assert "Hemorrhagic:         YES" in out


def test_present_negative_skips_stage2(capsys):
    present_case(_negative_trace(), pause=False)
    out = capsys.readouterr().out
    assert "STAGE 2 SKIPPED" in out
    assert "Stage 2 prompt" not in out  # Step 5 must not appear
    assert "Hemorrhagic:         NO" in out
    assert CLASS_NON_HAEMORRHAGIC in out


def test_present_positive_shows_cited_passages(capsys):
    present_case(_positive_trace(), pause=False)
    out = capsys.readouterr().out
    assert "CITED PASSAGES" in out
    assert "Eingeblutetes Kavernom" in out
    assert "Prior bleed still relevant" in out
    assert "Report 1 ·" in out
    assert "[Diagnosen]" in out


def test_split_report_sections():
    text = "[Diagnosen]\nFoo\n\n[Vorgehen/Beurteilung]\nBar"
    sections = _split_report_sections(text)
    assert sections == [("Diagnosen", "Foo"), ("Vorgehen/Beurteilung", "Bar")]


def test_evidenz_from_stage_raw_fallback():
    stage = {
        "parsed": {},
        "raw_response": '{"evidenz": [{"textstelle": "Blutung", "interpretation": "acute"}]}',
    }
    items = _evidenz_from_stage(stage)
    assert len(items) == 1
    assert items[0]["textstelle"] == "Blutung"


def test_render_cited_passages_empty_stage1(capsys):
    _render_cited_passages([], stage_label="Stage 1", compact_stage=True)
    out = capsys.readouterr().out
    assert "compact output" in out


def test_present_collapses_evidence_in_user_prompt(capsys):
    present_case(_positive_trace(), pause=False)
    out = capsys.readouterr().out
    # The clinical text is shown in Step 2 and collapsed inside the Step 3 user prompt.
    assert "clinical text — shown in STEP 2" in out


def test_collapse_evidence_replaces_text():
    user = "Header\n---\nLONG CLINICAL TEXT\n---\nReminder"
    collapsed = _collapse_evidence(user, "LONG CLINICAL TEXT")
    assert "LONG CLINICAL TEXT" not in collapsed
    assert "shown in STEP 2" in collapsed


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_autopick_negative():
    preds = pd.DataFrame(
        [
            {"case_id": "p", "status": "success", "label": "hämorrhagisch", "haemorrhage_subtype": "akut"},
            {"case_id": "n", "status": "success", "label": "nicht_hämorrhagisch", "haemorrhage_subtype": ""},
        ]
    ).astype(str)
    assert _autopick_negative_from_predictions(preds) == "n"


def test_autopick_negative_prefers_true_negative_over_false_negative():
    # 'fn' is predicted negative but is truly hemorrhagic (a mistake) → must be avoided.
    # 'tn' is predicted negative and truly non-hemorrhagic → must be chosen.
    preds = pd.DataFrame(
        [
            {
                "case_id": "fn",
                "status": "success",
                "label": "nicht_hämorrhagisch",
                "haemorrhage_subtype": "",
                "reference_label_status": "hemorrhagic",
            },
            {
                "case_id": "tn",
                "status": "success",
                "label": "nicht_hämorrhagisch",
                "haemorrhage_subtype": "",
                "reference_label_status": "non_hemorrhagic",
            },
        ]
    ).astype(str)
    assert _autopick_negative_from_predictions(preds) == "tn"


def test_autopick_negative_refuses_when_only_false_negative_available():
    # Reference labels exist but the only predicted-negative case is wrong → return None
    # rather than knowingly presenting a misclassification.
    preds = pd.DataFrame(
        [
            {
                "case_id": "fn",
                "status": "success",
                "label": "nicht_hämorrhagisch",
                "haemorrhage_subtype": "",
                "reference_label_status": "hemorrhagic",
            },
        ]
    ).astype(str)
    assert _autopick_negative_from_predictions(preds) is None


def test_autopick_negative_skips_excluded_pid():
    preds = pd.DataFrame(
        [
            {
                "case_id": "bad",
                "excel_pid": "10206120",
                "status": "success",
                "label": "nicht_hämorrhagisch",
                "haemorrhage_subtype": "",
                "reference_label_status": "non_hemorrhagic",
            },
            {
                "case_id": "good",
                "excel_pid": "99999",
                "status": "success",
                "label": "nicht_hämorrhagisch",
                "haemorrhage_subtype": "",
                "reference_label_status": "non_hemorrhagic",
            },
        ]
    ).astype(str)
    # Without exclusion the first TN ("bad") would be chosen; excluding its pid skips it.
    assert _autopick_negative_from_predictions(preds) == "bad"
    assert (
        _autopick_negative_from_predictions(preds, frozenset({"10206120"})) == "good"
    )


def test_autopick_positive_prefers_true_positive():
    from src.tasks.hemorrhage.demo_extraction import _autopick_from_predictions

    preds = pd.DataFrame(
        [
            {
                "case_id": "fp",
                "status": "success",
                "label": "hämorrhagisch",
                "haemorrhage_subtype": "akut",
                "reference_label_status": "non_hemorrhagic",
            },
            {
                "case_id": "tp",
                "status": "success",
                "label": "hämorrhagisch",
                "haemorrhage_subtype": "nicht_akut",
                "reference_label_status": "hemorrhagic",
            },
        ]
    ).astype(str)
    assert _autopick_from_predictions(preds) == "tp"


def test_select_polarity_falls_back_to_keyword_without_preds():
    cases = [
        _make_case("c_pos", "Akute Blutung mit Hämatom."),
        _make_case("c_neg", "Kavernom ohne Auffälligkeit, elektiv."),
    ]
    pos = _select_polarity_case(cases, None, kind="positive", case_id=None)
    neg = _select_polarity_case(cases, None, kind="negative", case_id=None)
    assert pos.case_id == "c_pos"
    assert neg.case_id == "c_neg"


def test_select_polarity_respects_case_id():
    cases = [_make_case("a", "x"), _make_case("b", "y")]
    assert _select_polarity_case(cases, None, kind="positive", case_id="b").case_id == "b"


# --------------------------------------------------------------------------- #
# Snapshot load round-trip
# --------------------------------------------------------------------------- #
def test_load_trace_round_trip(tmp_path, capsys):
    snap = tmp_path / "positive_case.json"
    snap.write_text(json.dumps(_positive_trace(), ensure_ascii=False), encoding="utf-8")
    loaded = load_trace(snap)
    assert loaded is not None
    present_case(loaded, pause=False)
    out = capsys.readouterr().out
    assert "STEP 7" in out


def test_load_trace_missing_returns_none(tmp_path):
    assert load_trace(tmp_path / "nope.json") is None
