"""Tests for hemorrhage prompt rules on preoperative vs remote historical bleeding."""

import json

from src.core.case.keys import CaseKey
from src.core.case.models import CaseReport, build_clinical_case
from src.tasks.hemorrhage.constants import (
    EXPECTED_REPORT_TYPUS_CODES,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.inference.parse import parse_hemorrhage_response
from src.tasks.hemorrhage.inference.prompt import (
    build_messages,
    build_user_prompt,
    load_system_prompt,
)


def _case_with_op_text(text: str):
    key = CaseKey("P1", "2024-01-01", "F1")
    reports = {
        TYPUS_OPERATIONSBERICHT: CaseReport(
            typus_code=TYPUS_OPERATIONSBERICHT,
            typus_label="01 Operationsbericht",
            report_text=text,
        )
    }
    return build_clinical_case(
        key,
        reports,
        expected_typus_codes=EXPECTED_REPORT_TYPUS_CODES,
    )


def test_system_prompt_contains_preoperative_bleeding_rule():
    prompt = load_system_prompt()
    assert "PRÄOPERATIVE BLUTUNG" in prompt
    assert "geblutetes Kavernom" in prompt
    assert "Hämatomevakuation" in prompt
    assert "Blutung 1998" in prompt


def test_preoperative_kavernom_prompt_includes_case_and_reminder():
    case = _case_with_op_text(
        "[Diagnosen]\ngeblutetes Kavernom\n\n[Vorgehen/Beurteilung]\nHämatomevakuation durchgeführt"
    )
    user = build_user_prompt(case)
    system = load_system_prompt()
    assert "geblutetes Kavernom" in user
    assert "Hämatomevakuation" in user
    assert "Präoperative Blutung" in user
    assert "geblutetes Kavernom" in system


def test_remote_history_prompt_includes_distinction():
    case = _case_with_op_text(
        "[Diagnosen]\nCCM\n\n[Indikation/Untersuch]\nBlutung 1998 in der Vorgeschichte, aktuell elektive Kontrolle"
    )
    user = build_user_prompt(case)
    assert "Blutung 1998" in user
    assert "ferne Vorgeschichte" in user


def test_example_response_preoperative_kavernom_class_1():
    """Expected model output shape for geblutetes Kavernom + Hämatomevakuation."""
    raw = json.dumps(
        {
            "klasse": 1,
            "label": "hämorrhagisch",
            "sicherheit": "hoch",
            "begruendung": "Geblutetes Kavernom war OP-Indikation; Hämatomevakuation behandelt präoperative Blutung im aktuellen Fall.",
            "evidenz": [
                {
                    "berichttyp": "01 Operationsbericht",
                    "feld": "diag",
                    "textstelle": "geblutetes Kavernom",
                    "interpretation": "Blutung im aktuellen Fallkontext, Indikation für OP",
                }
            ],
            "historische_blutung_erwaehnt": True,
            "historische_blutung_als_aktuell_gewertet": True,
            "unsicherheitsgruende": [],
        },
        ensure_ascii=False,
    )
    pred, err = parse_hemorrhage_response(raw, context="test_preop_kavernom")
    assert err is None
    assert pred["klasse"] == 1
    assert pred["historische_blutung_erwaehnt"] is True
    assert pred["historische_blutung_als_aktuell_gewertet"] is True


def test_example_response_remote_history_class_0():
    """Expected model output shape for remote Blutung 1998 only."""
    raw = json.dumps(
        {
            "klasse": 0,
            "label": "nicht_hämorrhagisch",
            "sicherheit": "mittel",
            "begruendung": "Blutung 1998 nur ferne Vorgeschichte ohne Bezug zur aktuellen Behandlung.",
            "evidenz": [],
            "historische_blutung_erwaehnt": True,
            "historische_blutung_als_aktuell_gewertet": False,
            "unsicherheitsgruende": [],
        },
        ensure_ascii=False,
    )
    pred, err = parse_hemorrhage_response(raw, context="test_remote_history")
    assert err is None
    assert pred["klasse"] == 0
    assert pred["historische_blutung_erwaehnt"] is True
    assert pred["historische_blutung_als_aktuell_gewertet"] is False


def test_build_messages_includes_updated_system_prompt():
    case = _case_with_op_text("geblutetes Kavernom, Hämatomevakuation")
    messages = build_messages(case)
    assert messages[0]["role"] == "system"
    assert "PRÄOPERATIVE BLUTUNG" in messages[0]["content"]
    assert "Hämatomevakuation" in messages[1]["content"]
