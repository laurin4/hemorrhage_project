"""Parse and validate hemorrhage case-level LLM JSON responses (stdlib only)."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VALID_SICHERHEIT = frozenset({"niedrig", "mittel", "hoch", "unbekannt"})

VALID_HAEMORRHAGE_SUBTYPES = frozenset({"akut", "historisch", "nicht_akut"})
SUBTYPE_UNKNOWN = "unbekannt"

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
    parse_repair_applied: str = ""


def _empty_prediction() -> Dict[str, Any]:
    return {
        "klasse": None,
        "label": "",
        "haemorrhage_subtype": None,
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


def _ok(pred: Dict[str, Any], *, parse_repair_applied: str = "") -> HemorrhageParseResult:
    return HemorrhageParseResult(
        prediction=pred,
        success=True,
        parse_repair_applied=parse_repair_applied,
    )


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


def sanitize_json_control_chars_in_strings(json_text: str) -> str:
    """
    Escape raw control characters that appear inside JSON string values.

    Walks the text character-by-character, tracks string context and escapes,
    and only modifies characters inside double-quoted strings.
    """
    if not json_text:
        return json_text

    out: List[str] = []
    in_string = False
    escape = False

    for ch in json_text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True
            escape = False

    return "".join(out)


def _try_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """
    Parse JSON object text. Returns (dict|None, error_detail, parse_repair_applied).

    Tries raw text first, then sanitized control-char repair inside strings.
    """
    if not text or not text.strip():
        return None, "empty text", ""

    raw = text.strip()
    candidates: List[Tuple[str, str]] = [(raw, "")]
    no_trail = _remove_trailing_commas(raw)
    if no_trail != raw:
        candidates.append((no_trail, ""))

    sanitized = sanitize_json_control_chars_in_strings(raw)
    if sanitized != raw:
        candidates.append((sanitized, "control_chars_escaped"))
        no_trail_sanitized = _remove_trailing_commas(sanitized)
        if no_trail_sanitized != sanitized:
            candidates.append((no_trail_sanitized, "control_chars_escaped"))

    last_err = ""
    seen: set[str] = set()
    for candidate, repair in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None, repair
            if isinstance(parsed, str):
                nested, nested_err, nested_repair = _try_json_loads(parsed)
                if nested is not None:
                    applied = repair or nested_repair
                    return nested, None, applied
                last_err = nested_err or "nested string not object"
        except json.JSONDecodeError as exc:
            last_err = str(exc)
    return None, last_err, ""


def _parse_llm_json_dict(
    raw_output: str, context: str
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """Extract and parse a JSON object from raw LLM text. Returns (dict|None, detail, repair)."""
    if not raw_output or not str(raw_output).strip():
        return None, "empty LLM response", ""

    text = _strip_bom(str(raw_output))
    text = _unescape_csv_style_quotes(text)
    text = _strip_markdown_fences(text)

    parsed, err, repair = _try_json_loads(text)
    if parsed is not None:
        return parsed, "", repair

    snippet = extract_first_json_object(text)
    if snippet:
        parsed, err, repair = _try_json_loads(snippet)
        if parsed is not None:
            return parsed, "", repair

    return None, err or f"no JSON object found ({context})", ""


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


def normalize_haemorrhage_subtype(value: object) -> Optional[str]:
    """
    Map subtype variants to canonical akut / historisch / nicht_akut.

    Returns None if the value is empty/null/unrecognized (caller decides default).
    """
    if value is None:
        return None
    key = _normalize_label_key(str(value))
    if not key or key in ("nan", "none", "null", "na"):
        return None

    # nicht_akut variants: nicht akut, nicht-akut, non_acute, chronisch, chronic, subakut(?)
    if key in ("nicht_akut", "non_acute", "nonacute", "chronisch", "chronic", "chronisch_akut"):
        return "nicht_akut"
    if key.startswith("nicht_akut") or "non_acut" in key or "chronisch" in key or "chronic" in key:
        return "nicht_akut"

    # historisch variants: history, historical, historisch, alt, früher (frueher), remote, anamnestisch
    if (
        key.startswith("historisch")
        or key.startswith("historic")
        or key in ("history", "historical", "anamnestisch", "alt", "frueher", "remote", "past", "old", "vergangen")
    ):
        return "historisch"

    # akut variants: akut, acute, akut_subakut, subakut
    if key.startswith("akut") or key.startswith("acut") or key.startswith("subakut") or key.startswith("subacut"):
        return "akut"

    return None


def _resolve_haemorrhage_subtype(
    raw_subtype: object,
    label: Optional[str],
) -> Tuple[Optional[str], bool]:
    """
    Resolve subtype based on label.

    Returns (subtype_value, subtype_uncertain).
    - non_hemorrhagic → (None, False)
    - hemorrhagic with valid subtype → (subtype, False)
    - hemorrhagic with missing/invalid subtype → ("unbekannt", True)
    """
    if label == "nicht_hämorrhagisch":
        return None, False

    normalized = normalize_haemorrhage_subtype(raw_subtype)
    if normalized in VALID_HAEMORRHAGE_SUBTYPES:
        return normalized, False

    # hemorrhagic but subtype unknown/missing
    return SUBTYPE_UNKNOWN, True


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


def _parse_binary_core(
    raw_output: str, context: str
) -> Tuple[Optional[HemorrhageParseResult], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Shared binary (klasse/label + reasoning) parsing for both single-call and
    two-stage flows.

    Returns ``(failure_result, out_prediction, parsed_dict, repair)``.
    - On failure: ``(HemorrhageParseResult, None, None, "")``.
    - On success: ``(None, out, parsed, repair)`` with ``out["haemorrhage_subtype"]``
      left as ``None`` (subtype resolution is the caller's responsibility).
    """
    if not raw_output or not str(raw_output).strip():
        return _fail("empty_llm_response", "empty LLM response"), None, None, ""

    try:
        parsed, detail, repair = _parse_llm_json_dict(raw_output, context)
    except Exception as exc:
        return _fail("unexpected_exception", str(exc)), None, None, ""

    if parsed is None:
        reason = "json_decode_error" if detail and "json" in detail.lower() else "no_json_object_found"
        if "no JSON object" in detail or "no json object" in detail.lower():
            reason = "no_json_object_found"
        if "Invalid control character" in detail:
            reason = "json_decode_error"
        return _fail(reason, detail), None, None, ""

    klasse, label, conflict = _resolve_klasse_label(parsed)
    if klasse is None or label is None:
        return _fail("missing_prediction_fields", conflict or detail), None, None, ""

    if conflict:
        partial = _empty_prediction()
        partial["klasse"] = klasse
        partial["label"] = label
        return (
            _fail("invalid_klasse_label_combination", conflict, partial=partial),
            None,
            None,
            "",
        )

    out = _empty_prediction()
    out["klasse"] = klasse
    out["label"] = label
    out["sicherheit"] = _normalize_sicherheit(parsed.get("sicherheit"))
    # Stage 1 (binary) uses compact "kurzbegruendung"; combined/legacy uses "begruendung".
    out["begruendung"] = str(
        parsed.get("begruendung") or parsed.get("kurzbegruendung") or ""
    ).strip()
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
    return None, out, parsed, repair


def parse_hemorrhage_response(raw_output: str, *, context: str) -> HemorrhageParseResult:
    """
    Parse a single combined LLM output → normalized prediction dict.

    Success requires extractable klasse + label. Optional fields use defaults.
    Resolves haemorrhage_subtype from the same payload (single-call mode).
    """
    fail, out, parsed, repair = _parse_binary_core(raw_output, context)
    if fail is not None:
        return fail

    subtype, subtype_uncertain = _resolve_haemorrhage_subtype(
        parsed.get("haemorrhage_subtype"), out["label"]
    )
    out["haemorrhage_subtype"] = subtype
    if subtype_uncertain:
        reasons = out["unsicherheitsgruende"]
        note = "haemorrhage_subtype fehlt oder unklar (auf 'unbekannt' gesetzt)"
        if note not in reasons:
            reasons.append(note)
        out["unsicherheitsgruende"] = reasons

    return _ok(out, parse_repair_applied=repair)


def parse_binary_response(raw_output: str, *, context: str) -> HemorrhageParseResult:
    """
    Parse Stage 1 (binary) LLM output → prediction dict WITHOUT subtype.

    ``haemorrhage_subtype`` is left as ``None``; the subtype is decided in
    Stage 2 (only for klasse=1). Same success/failure semantics as the combined
    parser.
    """
    fail, out, _parsed, repair = _parse_binary_core(raw_output, context)
    if fail is not None:
        return fail
    out["haemorrhage_subtype"] = None
    return _ok(out, parse_repair_applied=repair)


@dataclass
class SubtypeParseResult:
    """Structured Stage 2 (subtype) parse outcome."""

    haemorrhage_subtype: Optional[str] = None
    sicherheit: str = "unbekannt"
    begruendung: str = ""
    evidenz: List[dict] = field(default_factory=list)
    unsicherheitsgruende: List[str] = field(default_factory=list)
    success: bool = False
    subtype_uncertain: bool = False
    error_message: str = ""
    parse_error_reason: str = ""
    parse_error_detail: str = ""
    parse_repair_applied: str = ""


def parse_subtype_response(raw_output: str, *, context: str) -> SubtypeParseResult:
    """
    Parse Stage 2 (subtype-only) LLM output.

    Stage 2 assumes hemorrhage already exists; it only chooses
    akut / nicht_akut / historisch. A failed/unparseable response does not
    crash the pipeline: the subtype falls back to ``unbekannt`` and the caller
    records the uncertainty. ``success`` reflects whether a usable JSON object
    with a recognized subtype was produced.
    """
    if not raw_output or not str(raw_output).strip():
        return SubtypeParseResult(
            haemorrhage_subtype=SUBTYPE_UNKNOWN,
            subtype_uncertain=True,
            success=False,
            error_message="empty_llm_response: empty LLM response",
            parse_error_reason="empty_llm_response",
        )

    try:
        parsed, detail, repair = _parse_llm_json_dict(raw_output, context)
    except Exception as exc:
        return SubtypeParseResult(
            haemorrhage_subtype=SUBTYPE_UNKNOWN,
            subtype_uncertain=True,
            success=False,
            error_message=f"unexpected_exception: {exc}",
            parse_error_reason="unexpected_exception",
            parse_error_detail=str(exc)[:2000],
        )

    if parsed is None:
        reason = "json_decode_error" if detail and "json" in detail.lower() else "no_json_object_found"
        if "no JSON object" in detail or "no json object" in detail.lower():
            reason = "no_json_object_found"
        if "Invalid control character" in detail:
            reason = "json_decode_error"
        return SubtypeParseResult(
            haemorrhage_subtype=SUBTYPE_UNKNOWN,
            subtype_uncertain=True,
            success=False,
            error_message=f"{reason}: {detail}" if detail else reason,
            parse_error_reason=reason if reason in _PARSE_ERROR_REASONS else "unexpected_exception",
            parse_error_detail=detail[:2000] if detail else "",
        )

    normalized = normalize_haemorrhage_subtype(parsed.get("haemorrhage_subtype"))
    sicherheit = _normalize_sicherheit(parsed.get("sicherheit"))
    begruendung = str(parsed.get("begruendung", "") or "").strip()
    evidenz = _normalize_evidenz(parsed.get("evidenz"))
    unsicherheit = _normalize_unsicherheitsgruende(parsed.get("unsicherheitsgruende"))

    if normalized in VALID_HAEMORRHAGE_SUBTYPES:
        return SubtypeParseResult(
            haemorrhage_subtype=normalized,
            sicherheit=sicherheit,
            begruendung=begruendung,
            evidenz=evidenz,
            unsicherheitsgruende=unsicherheit,
            success=True,
            subtype_uncertain=False,
            parse_repair_applied=repair,
        )

    # JSON parsed but subtype missing/unrecognized → fall back to unbekannt.
    return SubtypeParseResult(
        haemorrhage_subtype=SUBTYPE_UNKNOWN,
        sicherheit=sicherheit,
        begruendung=begruendung,
        evidenz=evidenz,
        unsicherheitsgruende=unsicherheit,
        success=False,
        subtype_uncertain=True,
        error_message="haemorrhage_subtype fehlt oder unklar",
        parse_error_reason="missing_prediction_fields",
        parse_repair_applied=repair,
    )


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
