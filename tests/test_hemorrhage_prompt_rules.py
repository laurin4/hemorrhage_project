"""Tests for the two-level hemorrhage prompt rules and example response parsing.

Conceptual rule (current): historical hemorrhage is still hemorrhage, i.e.
klasse=1 / hämorrhagisch + haemorrhage_subtype="historisch". Historical bleeding
must NOT be classified as klasse=0.
"""

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


def _response(**overrides):
    base = {
        "klasse": 1,
        "label": "hämorrhagisch",
        "haemorrhage_subtype": "akut",
        "sicherheit": "hoch",
        "begruendung": "...",
        "evidenz": [],
        "historische_blutung_erwaehnt": False,
        "historische_blutung_als_aktuell_gewertet": False,
        "unsicherheitsgruende": [],
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# --- Prompt content -------------------------------------------------------


def test_system_prompt_requests_two_level_classification():
    prompt = load_system_prompt()
    assert "hämorrhagisch" in prompt
    assert "nicht_hämorrhagisch" in prompt
    assert "haemorrhage_subtype" in prompt
    for subtype in ("akut", "historisch", "nicht_akut"):
        assert subtype in prompt


def test_system_prompt_states_historical_is_hemorrhagic():
    prompt = load_system_prompt()
    lowered = prompt.lower()
    # New rule: historical hemorrhage is still hemorrhage (klasse=1 + historisch).
    assert "historische blutung ist weiterhin eine blutung" in lowered
    assert "niemals" in lowered
    # Old contradictory rule must be gone.
    assert "ferne Vorgeschichte" not in prompt
    assert "PRÄOPERATIVE BLUTUNG" not in prompt


def test_system_prompt_states_verify_vaskulaer_is_metadata():
    prompt = load_system_prompt()
    assert "Verify_Vaskulär" in prompt
    lowered = prompt.lower()
    assert "keine klassifikationsklasse" in lowered or "keine klasse" in lowered
    assert "nicht beeinflussen" in lowered


def test_user_prompt_reminder_mentions_historical_positive():
    case = _case_with_op_text("[Diagnosen]\nStatus nach Blutung 1998")
    user = build_user_prompt(case)
    assert "historische Blutung" in user
    assert "historisch" in user


def test_build_messages_includes_updated_system_prompt():
    case = _case_with_op_text("geblutetes Kavernom")
    messages = build_messages(case)
    assert messages[0]["role"] == "system"
    assert "haemorrhage_subtype" in messages[0]["content"]
    assert "historische Blutung ist weiterhin eine Blutung" in messages[0]["content"]


# --- Example response parsing (subtype hierarchy) -------------------------


def test_historical_hemorrhage_parses_positive_with_subtype_historisch():
    raw = _response(
        klasse=1,
        label="hämorrhagisch",
        haemorrhage_subtype="historisch",
        historische_blutung_erwaehnt=True,
        historische_blutung_als_aktuell_gewertet=False,
        begruendung="Status nach Blutung 1998.",
    )
    result = parse_hemorrhage_response(raw, context="hist")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["label"] == "hämorrhagisch"
    assert result.prediction["haemorrhage_subtype"] == "historisch"


def test_acute_hemorrhage_parses_positive_with_subtype_akut():
    raw = _response(klasse=1, haemorrhage_subtype="akut", begruendung="akute Einblutung")
    result = parse_hemorrhage_response(raw, context="akut")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["haemorrhage_subtype"] == "akut"


def test_current_non_acute_lesion_parses_positive_with_subtype_nicht_akut():
    raw = _response(
        klasse=1, haemorrhage_subtype="nicht_akut", begruendung="chronische hämorrhagische Läsion"
    )
    result = parse_hemorrhage_response(raw, context="na")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["haemorrhage_subtype"] == "nicht_akut"


def test_no_hemorrhage_evidence_parses_negative_with_null_subtype():
    raw = _response(
        klasse=0,
        label="nicht_hämorrhagisch",
        haemorrhage_subtype=None,
        begruendung="Kavernom ohne Einblutung.",
    )
    result = parse_hemorrhage_response(raw, context="neg")
    assert result.success
    assert result.prediction["klasse"] == 0
    assert result.prediction["haemorrhage_subtype"] is None


def test_remote_history_text_supports_historical_positive_response():
    """Remote history bleeding is hemorrhagic + historisch (not klasse 0)."""
    raw = _response(
        klasse=1,
        haemorrhage_subtype="historisch",
        begruendung="Frühere Blutung vor Jahren; aktuell keine akute Blutung.",
        historische_blutung_erwaehnt=True,
        historische_blutung_als_aktuell_gewertet=False,
    )
    result = parse_hemorrhage_response(raw, context="remote")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["haemorrhage_subtype"] == "historisch"
