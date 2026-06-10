"""Tests for the two-stage hierarchical hemorrhage inference architecture.

Stage 1 (binary): hämorrhagisch vs. nicht_hämorrhagisch.
Stage 2 (subtype): only for klasse=1 — akut / nicht_akut / historisch.
"""

import json
from pathlib import Path

import pandas as pd

from src.core.case.keys import CaseKey
from src.core.case.models import CaseReport, build_clinical_case
from src.tasks.hemorrhage.constants import (
    EXPECTED_REPORT_TYPUS_CODES,
    TYPUS_OPERATIONSBERICHT,
)
from src.tasks.hemorrhage.inference.parse import (
    parse_binary_response,
    parse_subtype_response,
)
from src.tasks.hemorrhage.inference.prompt import (
    build_binary_messages,
    build_subtype_messages,
    load_binary_system_prompt,
    load_subtype_system_prompt,
)
from src.tasks.hemorrhage.inference.runner import (
    PREDICTION_CSV_COLUMNS,
    _filter_to_cohort,
    process_single_case,
    run_hemorrhage_case_pipeline,
    write_predictions_csv,
)
from src.tasks.hemorrhage.io.reference_lookup import reference_binary_status


def _case(text: str):
    key = CaseKey("P1", "2024-01-01", "F1")
    reports = {
        TYPUS_OPERATIONSBERICHT: CaseReport(
            typus_code=TYPUS_OPERATIONSBERICHT,
            typus_label="01 Operationsbericht",
            report_text=text,
        )
    }
    return build_clinical_case(
        key, reports, expected_typus_codes=EXPECTED_REPORT_TYPUS_CODES
    )


def _binary_json(klasse: int, label: str) -> str:
    return json.dumps(
        {
            "klasse": klasse,
            "label": label,
            "sicherheit": "hoch",
            "begruendung": "binäre Begründung",
            "evidenz": [{"berichttyp": "01 Operationsbericht", "feld": "diag",
                         "textstelle": "x", "interpretation": "y"}],
            "historische_blutung_erwaehnt": False,
            "historische_blutung_als_aktuell_gewertet": False,
            "unsicherheitsgruende": [],
        },
        ensure_ascii=False,
    )


def _subtype_json(subtype: str) -> str:
    return json.dumps(
        {
            "haemorrhage_subtype": subtype,
            "sicherheit": "mittel",
            "begruendung": "subtyp Begründung",
            "evidenz": [{"berichttyp": "01 Operationsbericht", "feld": "diag",
                         "textstelle": "s", "interpretation": "t"}],
            "unsicherheitsgruende": [],
        },
        ensure_ascii=False,
    )


def _is_subtype_stage(messages) -> bool:
    """Stage 2 system prompt is the only one whose schema contains the
    ``haemorrhage_subtype`` output field."""
    return "haemorrhage_subtype" in messages[0]["content"]


def _staged_llm(binary_raw: str, subtype_raw: str = ""):
    """Fake llm_call that routes by which stage system prompt is present."""
    calls = {"binary": 0, "subtype": 0}

    def _call(messages):
        if _is_subtype_stage(messages):
            calls["subtype"] += 1
            return subtype_raw
        calls["binary"] += 1
        return binary_raw

    return _call, calls


# --- Stage prompts -----------------------------------------------------------


def test_binary_prompt_is_binary_only():
    prompt = load_binary_system_prompt()
    assert "hämorrhagisch" in prompt
    assert "nicht_hämorrhagisch" in prompt
    # Stage 1 must not request a subtype decision and must not emit the
    # subtype output field.
    assert "haemorrhage_subtype" not in prompt
    assert "KEIN Subtyp" in prompt
    # Compact Stage 1 output uses kurzbegruendung, no evidence list.
    assert "kurzbegruendung" in prompt


def test_binary_prompt_keeps_historical_positive():
    prompt = load_binary_system_prompt().lower()
    assert "historische blutung ist weiterhin klasse=1" in prompt
    assert "niemals klasse=0" in prompt


def test_binary_prompt_states_cavernoma_alone_not_hemorrhagic():
    prompt = load_binary_system_prompt().lower()
    assert "kavernom allein ist nicht hämorrhagisch" in prompt


def test_subtype_prompt_assumes_hemorrhage_exists():
    prompt = load_subtype_system_prompt()
    for s in ("historisch", "nicht_akut", "akut"):
        assert s in prompt
    lowered = prompt.lower()
    assert "bereits" in lowered and "hämorrhagisch" in lowered
    assert "stelle" in lowered  # "Stelle das NICHT in Frage."


def test_subtype_prompt_separates_historisch_from_not_acute():
    """'historisch' must NOT simply mean 'not acute' — relevance is the key."""
    lowered = load_subtype_system_prompt().lower()
    assert "bedeutet nicht einfach" in lowered
    assert "klinische relevanz" in lowered or "klinisch relevant" in lowered


def test_subtype_prompt_has_explicit_decision_rule():
    """Background-only → historisch; acute → akut; else → nicht_akut."""
    lowered = load_subtype_system_prompt().lower()
    assert "entscheidungsregel" in lowered
    assert "hintergrund-anamnese" in lowered
    # Ordered fallback to nicht_akut when not background and not acute.
    assert "nein" in lowered and "nicht_akut" in lowered


def test_subtype_prompt_older_but_relevant_maps_to_nicht_akut():
    """A bleed that is older but explains current symptoms/surgery → nicht_akut, not historisch."""
    prompt = load_subtype_system_prompt()
    lowered = prompt.lower()
    assert "eingeblutetes kavernom" in lowered
    assert "hämosiderin" in lowered
    # Explicit instruction that older-but-relevant is nicht_akut (not historisch).
    assert 'NICHT "historisch"' in prompt
    assert '"nicht_akut"' in prompt


# Behaviour expectations (subtype string → canonical normalization). The mapping
# from clinical text to subtype is the model's job; here we lock the canonical
# parsing for the labels the prompt is expected to emit for each scenario.
def test_subtype_parsing_for_documented_scenarios():
    scenarios = {
        # "St.n. Blutung 2010, unrelated to current surgery" → historisch
        "historisch": "historisch",
        # "eingeblutetes Kavernom caused current symptoms ... resection weeks later" → nicht_akut
        # "hemosiderin around current lesion relevant for treatment" → nicht_akut
        "nicht_akut": "nicht_akut",
        # "acute subdural hematoma with emergency evacuation" → akut
        "akut": "akut",
    }
    for expected, emitted in scenarios.items():
        res = parse_subtype_response(_subtype_json(emitted), context="t")
        assert res.success
        assert res.haemorrhage_subtype == expected


def test_build_stage_messages_distinct():
    case = _case("[Diagnosen]\nStatus nach Blutung 1998")
    b = build_binary_messages(case)
    s = build_subtype_messages(case)
    assert b[0]["content"] != s[0]["content"]
    assert "haemorrhage_subtype" in s[0]["content"]


# --- Stage parsers -----------------------------------------------------------


def test_parse_binary_leaves_subtype_none():
    res = parse_binary_response(_binary_json(1, "hämorrhagisch"), context="t")
    assert res.success
    assert res.prediction["klasse"] == 1
    assert res.prediction["haemorrhage_subtype"] is None


def test_parse_subtype_recognized():
    res = parse_subtype_response(_subtype_json("akut"), context="t")
    assert res.success
    assert res.haemorrhage_subtype == "akut"


def test_parse_subtype_missing_falls_back_unbekannt():
    res = parse_subtype_response(json.dumps({"sicherheit": "niedrig"}), context="t")
    assert not res.success
    assert res.haemorrhage_subtype == "unbekannt"
    assert res.subtype_uncertain


def test_parse_binary_accepts_compact_kurzbegruendung():
    raw = json.dumps(
        {
            "klasse": 0,
            "label": "nicht_hämorrhagisch",
            "sicherheit": "hoch",
            "kurzbegruendung": "Kavernom ohne Blutungsnachweis.",
        },
        ensure_ascii=False,
    )
    res = parse_binary_response(raw, context="t")
    assert res.success
    assert res.prediction["klasse"] == 0
    assert res.prediction["begruendung"] == "Kavernom ohne Blutungsnachweis."
    assert res.prediction["evidenz"] == []


# --- Cohort filtering --------------------------------------------------------


def test_reference_binary_status_mapping():
    def fields(h="", n="", v=""):
        return {
            "reference_haemorrhagisch": h,
            "reference_nicht_haemorrhagisch": n,
            "reference_verify_vaskulaer": v,
        }

    assert reference_binary_status(fields(h="ja")) == "hemorrhagic"
    assert reference_binary_status(fields(n="ja")) == "non_hemorrhagic"
    assert reference_binary_status(fields(v="ja")) == "verify_only"
    assert reference_binary_status(fields(h="ja", n="ja")) == "inconsistent"
    assert reference_binary_status(fields()) == "unknown"


def _cases_with_keys(keys):
    out = []
    for pid, opdat in keys:
        key = CaseKey(pid, opdat, f"F{pid}")
        reports = {
            TYPUS_OPERATIONSBERICHT: CaseReport(
                typus_code=TYPUS_OPERATIONSBERICHT,
                typus_label="01 Operationsbericht",
                report_text="text",
            )
        }
        out.append(
            build_clinical_case(key, reports, expected_typus_codes=EXPECTED_REPORT_TYPUS_CODES)
        )
    return out


def _lookup(entries):
    """entries: {(pid,opdat): (h,n,v)}"""
    return {
        (pid, opdat): {
            "reference_haemorrhagisch": h,
            "reference_nicht_haemorrhagisch": n,
            "reference_verify_vaskulaer": v,
        }
        for (pid, opdat), (h, n, v) in entries.items()
    }


def test_filter_to_cohort_default_keeps_only_binary_labeled():
    cases = _cases_with_keys([("1", "d"), ("2", "d"), ("3", "d"), ("4", "d"), ("5", "d")])
    lookup = _lookup(
        {
            ("1", "d"): ("ja", "", ""),   # hemorrhagic
            ("2", "d"): ("", "ja", ""),   # non_hemorrhagic
            ("3", "d"): ("", "", "ja"),   # verify_only
            ("4", "d"): ("ja", "ja", ""), # inconsistent
            ("5", "d"): ("", "", ""),     # unknown
        }
    )
    kept, excluded = _filter_to_cohort(cases, lookup, include_verify_only=False)
    kept_ids = {c.excel_pid for c in kept}
    assert kept_ids == {"1", "2"}
    assert excluded == {"verify_only": 1, "inconsistent": 1, "unknown": 1}


def test_filter_to_cohort_include_verify_only():
    cases = _cases_with_keys([("1", "d"), ("3", "d"), ("5", "d")])
    lookup = _lookup(
        {
            ("1", "d"): ("ja", "", ""),
            ("3", "d"): ("", "", "ja"),
            ("5", "d"): ("", "", ""),
        }
    )
    kept, excluded = _filter_to_cohort(cases, lookup, include_verify_only=True)
    assert {c.excel_pid for c in kept} == {"1", "3"}
    assert excluded == {"unknown": 1}


def test_pipeline_no_reference_processes_all_with_warning(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"excel_pid": "1", "excel_opdat": "2024-01-01", "opber_fallnr": "F1",
             "typus": "01 Operationsbericht", "diag": "kein Hinweis"},
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"
    call, _ = _staged_llm(_binary_json(0, "nicht_hämorrhagisch"))
    result = run_hemorrhage_case_pipeline(reports_path=xlsx, output_path=out, llm_call=call)
    # No reference available → cohort filter skipped, case still processed.
    assert result.cases_processed == 1
    assert "no reference available" in result.cohort_mode


def test_pipeline_all_cases_flag_skips_filter(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"excel_pid": "1", "excel_opdat": "2024-01-01", "opber_fallnr": "F1",
             "typus": "01 Operationsbericht", "diag": "kein Hinweis"},
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"
    call, _ = _staged_llm(_binary_json(0, "nicht_hämorrhagisch"))
    result = run_hemorrhage_case_pipeline(
        reports_path=xlsx, output_path=out, llm_call=call, process_all_cases=True
    )
    assert result.cases_processed == 1
    assert result.cohort_mode == "all_cases"


# --- Two-stage flow in process_single_case -----------------------------------


def test_klasse0_skips_subtype_stage():
    call, calls = _staged_llm(_binary_json(0, "nicht_hämorrhagisch"))
    row = process_single_case(_case("CCM ohne Einblutung"), {}, llm_call=call)
    assert row["status"] == "success"
    assert row["binary_stage_status"] == "success"
    assert row["subtype_stage_status"] == "skipped"
    assert row["klasse"] == 0
    assert row["haemorrhage_subtype"] == ""  # null subtype rendered as empty
    assert calls["binary"] == 1
    assert calls["subtype"] == 0


def test_klasse1_executes_subtype_stage():
    call, calls = _staged_llm(_binary_json(1, "hämorrhagisch"), _subtype_json("akut"))
    row = process_single_case(_case("akute Einblutung"), {}, llm_call=call)
    assert row["status"] == "success"
    assert row["binary_stage_status"] == "success"
    assert row["subtype_stage_status"] == "success"
    assert row["klasse"] == 1
    assert row["haemorrhage_subtype"] == "akut"
    assert calls["binary"] == 1
    assert calls["subtype"] == 1


def test_merged_output_combines_both_stages():
    call, _ = _staged_llm(_binary_json(1, "hämorrhagisch"), _subtype_json("historisch"))
    row = process_single_case(_case("Status nach Blutung 1998"), {}, llm_call=call)
    assert row["klasse"] == 1
    assert row["label"] == "hämorrhagisch"
    assert row["haemorrhage_subtype"] == "historisch"
    # Evidence from both stages is merged.
    evidenz = json.loads(row["evidenz_json"])
    assert len(evidenz) == 2
    # Binary + subtype reasoning combined.
    assert "Subtyp" in row["begruendung"]


def test_subtype_null_for_non_hemorrhagic():
    call, _ = _staged_llm(_binary_json(0, "nicht_hämorrhagisch"))
    row = process_single_case(_case("kein Blutungshinweis"), {}, llm_call=call)
    assert row["haemorrhage_subtype"] == ""


def test_subtype_populated_for_hemorrhagic():
    call, _ = _staged_llm(_binary_json(1, "hämorrhagisch"), _subtype_json("nicht_akut"))
    row = process_single_case(_case("residuelle Einblutung"), {}, llm_call=call)
    assert row["haemorrhage_subtype"] == "nicht_akut"


def test_subtype_stage_llm_failure_keeps_hemorrhagic():
    def call(messages):
        if _is_subtype_stage(messages):
            raise RuntimeError("subtype timeout")
        return _binary_json(1, "hämorrhagisch")

    row = process_single_case(_case("akute Blutung"), {}, llm_call=call)
    assert row["status"] == "success"
    assert row["binary_stage_status"] == "success"
    assert row["subtype_stage_status"] == "llm_failed"
    assert row["klasse"] == 1
    assert row["haemorrhage_subtype"] == "unbekannt"


def test_binary_stage_llm_failure_marks_llm_failed():
    def call(_messages):
        raise RuntimeError("binary timeout")

    row = process_single_case(_case("text"), {}, llm_call=call)
    assert row["status"] == "llm_failed"
    assert row["binary_stage_status"] == "llm_failed"


def test_binary_parse_failure_marks_parse_failed():
    call, calls = _staged_llm("not json at all")
    row = process_single_case(_case("text"), {}, llm_call=call)
    assert row["status"] == "parse_failed"
    assert row["binary_stage_status"] == "parse_failed"
    assert calls["subtype"] == 0


# --- CSV schema --------------------------------------------------------------


def test_csv_schema_unchanged_includes_subtype_and_stage_columns(tmp_path: Path):
    call, _ = _staged_llm(_binary_json(1, "hämorrhagisch"), _subtype_json("akut"))
    row = process_single_case(_case("akut"), {}, llm_call=call)
    write_predictions_csv([row], tmp_path / "out.csv")
    df = pd.read_csv(tmp_path / "out.csv")
    assert list(df.columns) == PREDICTION_CSV_COLUMNS
    for col in ("haemorrhage_subtype", "binary_stage_status", "subtype_stage_status",
                "binary_prompt_length", "subtype_prompt_length"):
        assert col in df.columns


def test_pipeline_two_stage_end_to_end(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"excel_pid": "1", "excel_opdat": "2024-01-01", "opber_fallnr": "F1",
             "typus": "01 Operationsbericht", "diag": "akute Blutung"},
            {"excel_pid": "2", "excel_opdat": "2024-01-02", "opber_fallnr": "F2",
             "typus": "01 Operationsbericht", "diag": "kein Hinweis"},
        ]
    )
    xlsx = tmp_path / "reports.xlsx"
    df.to_excel(xlsx, index=False)
    out = tmp_path / "preds.csv"

    def call(messages):
        if _is_subtype_stage(messages):
            return _subtype_json("akut")
        # First case hemorrhagic, second non-hemorrhagic, decide by case text.
        user = messages[1]["content"]
        if "akute Blutung" in user:
            return _binary_json(1, "hämorrhagisch")
        return _binary_json(0, "nicht_hämorrhagisch")

    result = run_hemorrhage_case_pipeline(reports_path=xlsx, output_path=out, llm_call=call)
    assert result.cases_processed == 2
    assert result.success_count == 2
    saved = pd.read_csv(out)
    statuses = set(saved["subtype_stage_status"].fillna("").tolist())
    assert "success" in statuses
    assert "skipped" in statuses
