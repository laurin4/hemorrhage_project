"""Parse and validate hemorrhage case-level LLM JSON responses (stdlib only)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

VALID_LABELS = frozenset({"nicht_hämorrhagisch", "hämorrhagisch", "nicht_haemorrhagisch", "haemorrhagisch"})
VALID_SICHERHEIT = frozenset({"niedrig", "mittel", "hoch"})

_USZ_TOKENS_TO_STRIP = (
    "<start_of_turn>user",
    "<start_of_turn>model",
    "<start_of_turn>",
    "<end_of_turn>",
)


def _strip_usz_template_tokens(text: str) -> str:
    out = text
    for tok in _USZ_TOKENS_TO_STRIP:
        out = out.replace(tok, "")
    return out


def extract_first_json_object(text: str) -> str:
    """
    Return the first complete top-level JSON object substring, or stripped original text.

    Uses brace matching with awareness of double-quoted strings and backslash escapes.
    """
    if text is None:
        return ""
    original_stripped = text.strip()
    if not original_stripped:
        return ""

    cleaned = _strip_usz_template_tokens(original_stripped).strip()

    start = cleaned.find("{")
    if start < 0:
        return original_stripped

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()

    return original_stripped


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_json_dict(raw_output: str, context: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from raw LLM text."""
    if not raw_output or not str(raw_output).strip():
        raise ValueError(f"empty LLM response ({context})")

    text = _strip_markdown_fences(str(raw_output))

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    snippet = extract_first_json_object(text)
    if snippet and snippet.startswith("{"):
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"json decode failed ({context}): {exc}") from exc

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"json decode failed ({context}): {exc}") from exc

    raise ValueError(f"no JSON object found ({context})")


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
        parsed = _parse_llm_json_dict(raw_output, context)
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
