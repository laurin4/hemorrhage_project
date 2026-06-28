"""Tests for the LLM extraction demo helpers (no live LLM / no raw data needed)."""

from __future__ import annotations

import json

import pandas as pd

from src.core.case.keys import CaseKey
from src.core.case.models import CaseReport, ClinicalCase
from src.tasks.hemorrhage.demo_extraction import (
    SUBTYPE_STAGE_DELIMITER,
    _autopick_from_predictions,
    _split_replay,
    build_trace,
    render_trace,
)
from src.tasks.hemorrhage.export.classification_merge import (
    CLASS_HAEMORRHAGIC_ACUTE,
    CLASS_NON_HAEMORRHAGIC,
)


def _make_case(text: str) -> ClinicalCase:
    return ClinicalCase(
        case_key=CaseKey("1", "2014-05-03", "F1"),
        case_id="case_1__2014-05-03__F1",
        reports={"01": CaseReport(typus_code="01", typus_label="01 Operationsbericht", report_text=text)},
        available_report_types=("01",),
    )


def test_split_replay_with_subtype_stage():
    raw = f'{{"klasse": 1}}\n\n{SUBTYPE_STAGE_DELIMITER}\n\n{{"haemorrhage_subtype": "akut"}}'
    binary_raw, subtype_raw = _split_replay(raw)
    assert binary_raw == '{"klasse": 1}'
    assert subtype_raw == '{"haemorrhage_subtype": "akut"}'


def test_split_replay_without_subtype_stage():
    raw = '{"klasse": 0, "label": "nicht_hämorrhagisch"}'
    binary_raw, subtype_raw = _split_replay(raw)
    assert binary_raw == raw
    assert subtype_raw == ""


def test_autopick_prefers_hemorrhagic_with_subtype():
    preds = pd.DataFrame(
        [
            {"case_id": "c0", "status": "success", "label": "nicht_hämorrhagisch", "haemorrhage_subtype": ""},
            {"case_id": "c1", "status": "success", "label": "hämorrhagisch", "haemorrhage_subtype": "akut"},
        ]
    ).astype(str)
    assert _autopick_from_predictions(preds) == "c1"


def test_autopick_returns_none_when_no_hemorrhagic():
    preds = pd.DataFrame(
        [{"case_id": "c0", "status": "success", "label": "nicht_hämorrhagisch", "haemorrhage_subtype": ""}]
    ).astype(str)
    assert _autopick_from_predictions(preds) is None


def test_build_trace_hemorrhagic_runs_both_stages():
    case = _make_case("Akute intrazerebrale Blutung, Hämatomevakuation.")
    binary_raw = (
        '{"klasse": 1, "label": "hämorrhagisch", "sicherheit": "hoch", '
        '"kurzbegruendung": "akute Blutung"}'
    )
    subtype_raw = (
        '{"haemorrhage_subtype": "akut", "sicherheit": "hoch", "begruendung": "frisch"}'
    )
    trace = build_trace(
        case, binary_raw=binary_raw, subtype_raw=subtype_raw, stage2_ran=True, mode="TEST"
    )
    assert trace["stage1"]["parsed"]["klasse"] == 1
    assert trace["stage2"] is not None
    assert trace["stage2"]["parsed"]["haemorrhage_subtype"] == "akut"
    assert trace["final"]["class_column"] == CLASS_HAEMORRHAGIC_ACUTE


def test_build_trace_non_hemorrhagic_skips_stage2():
    case = _make_case("Kavernom ohne Einblutung, elektive Resektion.")
    binary_raw = (
        '{"klasse": 0, "label": "nicht_hämorrhagisch", "sicherheit": "mittel", '
        '"kurzbegruendung": "keine Blutung"}'
    )
    trace = build_trace(
        case, binary_raw=binary_raw, subtype_raw="", stage2_ran=False, mode="TEST"
    )
    assert trace["stage2"] is None
    assert trace["final"]["class_column"] == CLASS_NON_HAEMORRHAGIC


def test_snapshot_round_trip_renders(tmp_path, capsys):
    case = _make_case("Akute Blutung mit Hämatom.")
    trace = build_trace(
        case,
        binary_raw='{"klasse": 1, "label": "hämorrhagisch", "sicherheit": "hoch", "kurzbegruendung": "Blutung"}',
        subtype_raw='{"haemorrhage_subtype": "akut", "sicherheit": "hoch", "begruendung": "frisch"}',
        stage2_ran=True,
        mode="TEST",
    )
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = json.loads(snap.read_text(encoding="utf-8"))
    render_trace(loaded, max_text_chars=None)
    out = capsys.readouterr().out
    assert "SCHRITT 1" in out
    assert "SCHRITT 7" in out
    assert CLASS_HAEMORRHAGIC_ACUTE in out
