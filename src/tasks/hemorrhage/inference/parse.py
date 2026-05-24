"""Parse and validate hemorrhage case-level LLM JSON responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.models.json_parsing import parse_llm_json_output
from src.models.llm_interface import extract_first_json_object

VALID_LABELS = frozenset({"nicht_hämorrhagisch", "hämorrhagisch", "nicht_haemorrhagisch", "haemorrhagisch"})
VALID_SICHERHEIT = frozenset({"niedrig", "mittel", "hoch"})


def _empty_prediction() -> Dict[str, Any]:
    return {
        "klasse": None,
        "label": "",
        "sicherheit": "",
        "begruendung": "",
        "evidenz": [],
        "historische_blutung_erwaehnt": None,
        "historische_blutung_als_aktuell_gewertet": None,
        "unsicherheitsgruende": [],
    }


def parse_hemorrhage_response(raw_output: str, *, context: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Parse LLM output → normalized prediction dict.

    Returns (prediction_fields, error_message). error_message is None on success.
    """
    if not raw_output or not str(raw_output).strip():
        return _empty_prediction(), "empty_llm_response"

    try:
        parsed = parse_llm_json_output(raw_output, context)
    except Exception:
        try:
            snippet = extract_first_json_object(raw_output)
            parsed = json.loads(snippet) if snippet else {}
        except Exception as exc:
            return _empty_prediction(), f"json_parse_failed: {exc}"

    if not isinstance(parsed, dict):
        return _empty_prediction(), "json_not_object"

    out = _empty_prediction()
    errors: List[str] = []

    klasse = parsed.get("klasse")
    try:
        if klasse is not None:
            ki = int(klasse)
            if ki not in (0, 1):
                errors.append("invalid_klasse")
            else:
                out["klasse"] = ki
    except (TypeError, ValueError):
        errors.append("invalid_klasse")

    label = str(parsed.get("label", "") or "").strip()
    if label:
        if label in VALID_LABELS:
            out["label"] = label.replace("haemorrhagisch", "hämorrhagisch").replace(
                "nicht_haemorrhagisch", "nicht_hämorrhagisch"
            )
        else:
            errors.append("invalid_label")
            out["label"] = label

    sicherheit = str(parsed.get("sicherheit", "") or "").strip().lower()
    if sicherheit:
        if sicherheit in VALID_SICHERHEIT:
            out["sicherheit"] = sicherheit
        else:
            errors.append("invalid_sicherheit")
            out["sicherheit"] = sicherheit

    out["begruendung"] = str(parsed.get("begruendung", "") or "").strip()

    evidenz = parsed.get("evidenz", [])
    if isinstance(evidenz, list):
        out["evidenz"] = [e for e in evidenz if isinstance(e, dict)]
    else:
        errors.append("invalid_evidenz")

    for bool_key in ("historische_blutung_erwaehnt", "historische_blutung_als_aktuell_gewertet"):
        val = parsed.get(bool_key)
        if val is None:
            continue
        if isinstance(val, bool):
            out[bool_key] = val
        elif str(val).strip().lower() in ("true", "1", "yes", "ja"):
            out[bool_key] = True
        elif str(val).strip().lower() in ("false", "0", "no", "nein"):
            out[bool_key] = False
        else:
            errors.append(f"invalid_{bool_key}")

    unsicher = parsed.get("unsicherheitsgruende", [])
    if isinstance(unsicher, list):
        out["unsicherheitsgruende"] = [str(u).strip() for u in unsicher if str(u).strip()]
    elif unsicher:
        errors.append("invalid_unsicherheitsgruende")

    if errors:
        return out, "validation: " + ",".join(errors)
    return out, None


def evidenz_to_json(evidenz: List[dict]) -> str:
    return json.dumps(evidenz, ensure_ascii=False)


def list_to_json(items: List[str]) -> str:
    return json.dumps(items, ensure_ascii=False)
