"""Robust hemorrhage LLM JSON parsing tests."""

import json

from src.tasks.hemorrhage.inference.parse import (
    parse_hemorrhage_response,
    parse_hemorrhage_response_legacy,
)


def _valid_payload(**overrides):
    base = {
        "klasse": 1,
        "label": "hämorrhagisch",
        "sicherheit": "hoch",
        "begruendung": "Aktuelle Blutung im OP-Bericht.",
        "evidenz": [
            {
                "berichttyp": "01 Operationsbericht",
                "feld": "diag",
                "textstelle": "frische Blutung",
                "interpretation": "aktuell relevant",
            }
        ],
        "historische_blutung_erwaehnt": False,
        "historische_blutung_als_aktuell_gewertet": False,
        "unsicherheitsgruende": [],
    }
    base.update(overrides)
    return base


def test_pure_valid_json_success():
    result = parse_hemorrhage_response(json.dumps(_valid_payload()), context="t1")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["label"] == "hämorrhagisch"


def test_json_in_markdown_fence_success():
    raw = "```json\n" + json.dumps(_valid_payload(), ensure_ascii=False) + "\n```"
    result = parse_hemorrhage_response(raw, context="t2")
    assert result.success
    assert result.prediction["klasse"] == 1


def test_json_with_leading_trailing_text_success():
    raw = "Analyse:\n" + json.dumps(_valid_payload()) + "\nEnde."
    result = parse_hemorrhage_response(raw, context="t3")
    assert result.success


def test_klasse_as_string_success():
    payload = _valid_payload(klasse="1")
    result = parse_hemorrhage_response(json.dumps(payload), context="t4")
    assert result.success
    assert result.prediction["klasse"] == 1


def test_label_only_fallback_success():
    payload = _valid_payload()
    del payload["klasse"]
    payload["label"] = "haemorrhagisch"
    result = parse_hemorrhage_response(json.dumps(payload), context="t5")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.prediction["label"] == "hämorrhagisch"


def test_invalid_evidence_but_valid_klasse_label_success():
    payload = _valid_payload(evidenz="not-a-list", sicherheit="very_high")
    result = parse_hemorrhage_response(json.dumps(payload), context="t6")
    assert result.success
    assert result.prediction["evidenz"] == []
    assert result.prediction["sicherheit"] == "unbekannt"


def test_missing_klasse_and_label_parse_failed():
    payload = _valid_payload()
    del payload["klasse"]
    del payload["label"]
    result = parse_hemorrhage_response(json.dumps(payload), context="t7")
    assert not result.success
    assert result.parse_error_reason == "missing_prediction_fields"


def test_malformed_json_parse_failed_with_reason():
    result = parse_hemorrhage_response("{klasse: 1", context="t8")
    assert not result.success
    assert result.parse_error_reason in ("json_decode_error", "no_json_object_found")


def test_typical_csv_style_response_with_evidence_success():
    payload = _valid_payload(
        label="hämorrhagisch",
        evidenz=[
            {
                "berichttyp": "03 Austrittsbericht",
                "feld": "diag",
                "textstelle": "Hämatomevakuation",
                "interpretation": "fallrelevant",
            }
        ],
    )
    raw = json.dumps(payload, ensure_ascii=False)
    result = parse_hemorrhage_response(raw, context="t9")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert len(result.prediction["evidenz"]) == 1


def test_doubled_csv_quotes_json_success():
    inner = json.dumps(_valid_payload(klasse=1, label="hämorrhagisch"))
    wrapped = '"' + inner.replace('"', '""') + '"'
    result = parse_hemorrhage_response(wrapped, context="t10")
    assert result.success


def test_label_variants_normalized():
    for label in ("hemorrhagisch", "nicht haemorrhagisch", "nicht_hemorrhagisch"):
        payload = _valid_payload(label=label)
        del payload["klasse"]
        if "nicht" in label:
            payload["klasse"] = 0
        result = parse_hemorrhage_response(json.dumps(payload), context="t11")
        assert result.success, label


def test_legacy_tuple_api():
    pred, err = parse_hemorrhage_response_legacy(
        json.dumps(_valid_payload()), context="legacy"
    )
    assert err is None
    assert pred["klasse"] == 1


def test_invalid_klasse_label_combination():
    payload = _valid_payload(klasse=0, label="hämorrhagisch")
    result = parse_hemorrhage_response(json.dumps(payload), context="t12")
    assert not result.success
    assert result.parse_error_reason == "invalid_klasse_label_combination"


def test_raw_newline_inside_string_repaired_success():
    raw = (
        '{"klasse": 1, "label": "hämorrhagisch", "begruendung": "erste zeile'
        + '\n'
        + 'zweite zeile", "evidenz": [{"berichttyp":"01 Operationsbericht","feld":"diag","textstelle":"text'
        + '\n'
        + 'fortsetzung"}]}'
    )
    result = parse_hemorrhage_response(raw, context="ctrl_nl")
    assert result.success
    assert result.prediction["klasse"] == 1
    assert result.parse_repair_applied == "control_chars_escaped"


def test_raw_tab_inside_string_repaired_success():
    raw = '{"klasse": 1, "label": "hämorrhagisch", "begruendung": "a' + '\t' + 'b"}'
    result = parse_hemorrhage_response(raw, context="ctrl_tab")
    assert result.success
    assert result.parse_repair_applied == "control_chars_escaped"


def test_valid_json_outside_string_formatting_still_parses():
    raw = '{\n  "klasse": 0,\n  "label": "nicht_hämorrhagisch",\n  "evidenz": []\n}'
    result = parse_hemorrhage_response(raw, context="fmt")
    assert result.success
    assert result.prediction["klasse"] == 0
    assert result.parse_repair_applied == ""


def test_nonrecoverable_malformed_json_still_fails():
    raw = '{"klasse": 1, "label": "hämorrhagisch",'
    result = parse_hemorrhage_response(raw, context="bad")
    assert not result.success
    assert result.parse_error_reason in ("json_decode_error", "no_json_object_found")


def test_subtype_akut_parsed():
    payload = _valid_payload(haemorrhage_subtype="akut")
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_akut")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] == "akut"


def test_subtype_historisch_parsed():
    payload = _valid_payload(haemorrhage_subtype="historisch")
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_hist")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] == "historisch"


def test_subtype_nicht_akut_parsed():
    payload = _valid_payload(haemorrhage_subtype="nicht_akut")
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_na")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] == "nicht_akut"


def test_subtype_normalizes_english_and_chronic_variants():
    for raw_val, expected in [
        ("acute", "akut"),
        ("historical", "historisch"),
        ("history", "historisch"),
        ("alt", "historisch"),
        ("früher", "historisch"),
        ("remote", "historisch"),
        ("non_acute", "nicht_akut"),
        ("non-acute", "nicht_akut"),
        ("nicht-akut", "nicht_akut"),
        ("chronisch", "nicht_akut"),
        ("chronic", "nicht_akut"),
    ]:
        payload = _valid_payload(haemorrhage_subtype=raw_val)
        result = parse_hemorrhage_response(json.dumps(payload), context="sub_norm")
        assert result.success
        assert result.prediction["haemorrhage_subtype"] == expected, raw_val


def test_subtype_null_allowed_for_non_hemorrhagic():
    payload = _valid_payload(
        klasse=0, label="nicht_hämorrhagisch", haemorrhage_subtype=None
    )
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_null")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] is None


def test_non_hemorrhagic_ignores_provided_subtype():
    payload = _valid_payload(
        klasse=0, label="nicht_hämorrhagisch", haemorrhage_subtype="akut"
    )
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_force_null")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] is None


def test_missing_subtype_for_hemorrhagic_does_not_crash():
    payload = _valid_payload()  # hämorrhagisch, no subtype field
    payload.pop("haemorrhage_subtype", None)
    result = parse_hemorrhage_response(json.dumps(payload), context="sub_missing")
    assert result.success
    assert result.prediction["haemorrhage_subtype"] == "unbekannt"
    assert any("haemorrhage_subtype" in r for r in result.prediction["unsicherheitsgruende"])
