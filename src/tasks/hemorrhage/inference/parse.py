"""Parse and validate hemorrhage case-level LLM JSON responses (stdlib only)."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VALID_SICHERHEIT = frozenset({"niedrig", "mittel", "hoch", "unbekannt"})

_USZ_TOKENS_TO_STRIP = (
    "<start_of_turn>user",
    "<start_of_turn>model",
    "<start_of_turn>",
    "<end_of_turn>",
)

_PARSE_ERROR_REASONS = frozenset(
    {
        "no_json_object_found",
        "json_decode_error",
        "missing_prediction_fields",
        "invalid_klasse_label_combination",
        "unexpected_exception",
        "empty_llm_response",
    }
)


@dataclass
class HemorrhageParseResult:
    """Structured parse outcome for pipeline and debug exports."""

    prediction: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: str = ""
    parse_error_reason: str = ""
    parse_error_detail: str = ""


def _empty_prediction() -> Dict[str, Any]:
    return {
        "klasse": None,
        "label": "",
        "sicherheit": "unbekannt",
        "begruendung": "",
        "evidenz": [],
        "historische_blutung_erwaehnt": None,
        "historische_blutung_als_aktuell_gewertet": None,
        "unsicherheitsgruende": [],
    }


def _fail(
    reason: str,
    detail: str = "",
    *,
    partial: Optional[Dict[str, Any]] = None,
) -> HemorrhageParseResult:
    pred = partial if partial is not None else _empty_prediction()
    msg = reason if not detail else f"{reason}: {detail}"
    return HemorrhageParseResult(
        prediction=pred,
        success=False,
        error_message=msg,
        parse_error_reason=reason if reason in _PARSE_ERROR_REASONS else "unexpected_exception",
        parse_error_detail=detail[:2000] if detail else "",
    )


def _ok(pred: Dict[str, Any]) -> HemorrhageParseResult:
    return HemorrhageParseResult(prediction=pred, success=True)


def _strip_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch) != "Cc")


def _strip_usz_template_tokens(text: str) -> str:
    out = text
    for tok in _USZ_TOKENS_TO_STRIP:
        out = out.replace(tok, "")
    return out


def _unescape_csv_style_quotes(text: str) -> str:
    """Unwrap doubled quotes from CSV-embedded JSON strings."""
    s = text.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1].replace('""', '"')
        if inner.lstrip().startswith("{"):
            return inner
    return text


def _strip_markdown_fences(text: str) -> str:
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _remove_trailing_commas(json_text: str) -> str:
    """Remove trailing commas before } or ] (common LLM mistake)."""
    return re.sub(r",(\s*[}\]])", r"\1", json_text)


def extract_first_json_object(text: str) -> str:
    """
    Return the first complete top-level JSON object substring.

    Uses brace matching with awareness of double-quoted strings and escapes.
    """
    if text is None:
        return ""
    original_stripped = text.strip()
    if not original_stripped:
        return ""

    cleaned = _strip_usz_template_tokens(original_stripped).strip()
    start = cleaned.find("{")
    if start < 0:
        return ""

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

    return ""


def _try_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "empty text"
    candidates = [text.strip(), _remove_trailing_commas(text.strip())]
    last_err = ""
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
            if isinstance(parsed, str):
                nested, nested_err = _try_json_loads(parsed)
                if nested is not None:
                    return nested, None
                last_err = nested_err or "nested string not object"
        except json.JSONDecodeError as exc:
            last_err = str(exc)
    return None, last_err


def _parse_llm_json_dict(raw_output: str, context: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract and parse a JSON object from raw LLM text. Returns (dict|None, detail)."""
    if not raw_output or not str(raw_output).strip():
        return None, "empty LLM response"

    text = _strip_bom(str(raw_output))
    text = _strip_control_chars(text)
    text = _unescape_csv_style_quotes(text)
    text = _strip_markdown_fences(text)

    parsed, err = _try_json_loads(text)
    if parsed is not None:
        return parsed, ""

    snippet = extract_first_json_object(text)
    if snippet:
        parsed, err = _try_json_loads(snippet)
        if parsed is not None:
            return parsed, ""

    return None, err or f"no JSON object found ({context})"


def _normalize_label_key(label: str) -> str:
    s = str(label or "").strip().lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def normalize_label(label: object) -> Optional[str]:
    """Map label variants to canonical hämorrhagisch / nicht_hämorrhagisch."""
    if label is None:
        return None
    key = _normalize_label_key(str(label))
    if not key:
        return None

    if key.startswith("nicht"):
        if any(token in key for token in ("haemorrhag", "hamorrhag", "hemorrhag")):
            return "nicht_hämorrhagisch"
        return None

    if any(token in key for token in ("haemorrhag", "hamorrhag", "hemorrhag")):
        return "hämorrhagisch"

    return None


def _parse_klasse_value(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            k = int(value)
            return k if k in (0, 1) else None
        except (TypeError, ValueError):
            return None
    s = str(value).strip().lower()
    if not s or s in ("nan", "none", "<na>"):
        return None
    if s in ("1", "true", "yes", "ja"):
        return 1
    if s in ("0", "false", "no", "nein"):
        return 0
    try:
        k = int(float(s))
        return k if k in (0, 1) else None
    except (TypeError, ValueError):
        return None


def _klasse_from_label(label: Optional[str]) -> Optional[int]:
    if label == "hämorrhagisch":
        return 1
    if label == "nicht_hämorrhagisch":
        return 0
    return None


def _label_from_klasse(klasse: Optional[int]) -> Optional[str]:
    if klasse == 1:
        return "hämorrhagisch"
    if klasse == 0:
        return "nicht_hämorrhagisch"
    return None


def _parse_bool_optional(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if not s or s in ("nan", "none", "<na>", "null"):
        return None
    if s in ("true", "1", "yes", "ja"):
        return True
    if s in ("false", "0", "no", "nein"):
        return False
    return None


def _normalize_evidenz(raw: object) -> List[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict):
        return [raw]
    return []


def _normalize_unsicherheitsgruende(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _normalize_sicherheit(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s in VALID_SICHERHEIT:
        return s
    return "unbekannt"


def _resolve_klasse_label(
    parsed: Dict[str, Any],
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Resolve klasse + canonical label with fallbacks.

    Returns (klasse, label, error_detail) — error_detail set on inconsistent pair.
    """
    klasse = _parse_klasse_value(parsed.get("klasse"))
    label = normalize_label(parsed.get("label"))

    if klasse is None and label is not None:
        klasse = _klasse_from_label(label)
    if label is None and klasse is not None:
        label = _label_from_klasse(klasse)

    if klasse is None or label is None:
        return None, None, "missing klasse and/or label"

    expected = _klasse_from_label(label)
    if expected is not None and expected != klasse:
        return klasse, label, f"klasse={klasse} conflicts with label={label}"

    return klasse, label, None


def parse_hemorrhage_response(raw_output: str, *, context: str) -> HemorrhageParseResult:
    """
    Parse LLM output → normalized prediction dict.

    Success requires extractable klasse + label. Optional fields use defaults.
    """
    if not raw_output or not str(raw_output).strip():
        return _fail("empty_llm_response", "empty LLM response")

    try:
        parsed, detail = _parse_llm_json_dict(raw_output, context)
    except Exception as exc:
        return _fail("unexpected_exception", str(exc))

    if parsed is None:
        reason = "json_decode_error" if detail and "json" in detail.lower() else "no_json_object_found"
        if "no JSON object" in detail or "no json object" in detail.lower():
            reason = "no_json_object_found"
        return _fail(reason, detail)

    klasse, label, conflict = _resolve_klasse_label(parsed)
    if klasse is None or label is None:
        return _fail("missing_prediction_fields", conflict or detail)

    if conflict:
        partial = _empty_prediction()
        partial["klasse"] = klasse
        partial["label"] = label
        return _fail("invalid_klasse_label_combination", conflict, partial=partial)

    out = _empty_prediction()
    out["klasse"] = klasse
    out["label"] = label
    out["sicherheit"] = _normalize_sicherheit(parsed.get("sicherheit"))
    out["begruendung"] = str(parsed.get("begruendung", "") or "").strip()
    out["evidenz"] = _normalize_evidenz(parsed.get("evidenz"))
    out["historische_blutung_erwaehnt"] = _parse_bool_optional(
        parsed.get("historische_blutung_erwaehnt")
    )
    out["historische_blutung_als_aktuell_gewertet"] = _parse_bool_optional(
        parsed.get("historische_blutung_als_aktuell_gewertet")
    )
    out["unsicherheitsgruende"] = _normalize_unsicherheitsgruende(
        parsed.get("unsicherheitsgruende")
    )

    return _ok(out)


def parse_hemorrhage_response_legacy(
    raw_output: str, *, context: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Backward-compatible (prediction, error_message) tuple."""
    result = parse_hemorrhage_response(raw_output, context=context)
    if result.success:
        return result.prediction, None
    return result.prediction, result.error_message


def evidenz_to_json(evidenz: List[dict]) -> str:
    return json.dumps(evidenz, ensure_ascii=False)


def list_to_json(items: List[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def preview_snippet(text: str, max_chars: int = 500) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars]
