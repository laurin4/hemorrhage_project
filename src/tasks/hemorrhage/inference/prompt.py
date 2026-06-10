"""
German system prompt for hemorrhage case-level classification.

Loads from ``prompts/hemorrhage_case_classification.txt`` relative to project root.
"""

from __future__ import annotations

from pathlib import Path

from src.core.case.models import ClinicalCase
from src.pipeline.paths import PROJECT_ROOT
from src.tasks.hemorrhage.constants import TYPUS_CODE_TO_LABEL

PROMPT_PATH = PROJECT_ROOT / "prompts" / "hemorrhage_case_classification.txt"
BINARY_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hemorrhage_binary_classification.txt"
SUBTYPE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "hemorrhage_subtype_classification.txt"

PROMPT_PREVIEW_MAX_CHARS = 2000


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM_PROMPT


def load_binary_system_prompt() -> str:
    """Stage 1 system prompt: binary hemorrhagic vs non-hemorrhagic only."""
    if BINARY_PROMPT_PATH.exists():
        return BINARY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _FALLBACK_BINARY_SYSTEM_PROMPT


def load_subtype_system_prompt() -> str:
    """Stage 2 system prompt: subtype only (hemorrhage already confirmed)."""
    if SUBTYPE_PROMPT_PATH.exists():
        return SUBTYPE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _FALLBACK_SUBTYPE_SYSTEM_PROMPT


def _format_available_reports(case: ClinicalCase) -> str:
    if case.available_report_types:
        labels = [TYPUS_CODE_TO_LABEL.get(c, c) for c in case.available_report_types]
        return ", ".join(labels)
    return "(keine typisierten Berichte vorhanden)"


def _format_missing_reports(case: ClinicalCase) -> str:
    if case.missing_report_types:
        labels = [TYPUS_CODE_TO_LABEL.get(c, c) for c in case.missing_report_types]
        return ", ".join(labels)
    return "(keine)"


def _case_block(case: ClinicalCase) -> str:
    """Shared case-context block (identifiers + reports + structured text)."""
    return f"""Fall-Identifikatoren:
- excel_pid: {case.excel_pid}
- excel_opdat: {case.excel_opdat}
- opber_fallnr: {case.opber_fallnr}
- case_id: {case.case_id}

Verfügbare Berichtstypen: {_format_available_reports(case)}
Fehlende Berichtstypen: {_format_missing_reports(case)}

Strukturierter Falltext (alle verfügbaren Berichte):
---
{case.structured_case_text() or "(kein Berichtstext verfügbar)"}
---"""


def build_user_prompt(case: ClinicalCase) -> str:
    """User message: case metadata + structured case text (combined single-call)."""
    return f"""{_case_block(case)}

{_USER_PROMPT_REMINDER}

Analysiere diesen klinischen Fall und gib NUR das JSON-Objekt gemäss Schema zurück.
"""


def build_binary_user_prompt(case: ClinicalCase) -> str:
    """Stage 1 user message: only ask for hemorrhagic vs non-hemorrhagic."""
    return f"""{_case_block(case)}

{_BINARY_USER_PROMPT_REMINDER}

Entscheide NUR: hämorrhagisch (klasse=1) oder nicht_hämorrhagisch (klasse=0).
Gib NUR das JSON-Objekt gemäss Schema zurück (kein Subtyp).
"""


def build_subtype_user_prompt(case: ClinicalCase) -> str:
    """Stage 2 user message: hemorrhage is confirmed, only ask for subtype."""
    return f"""{_case_block(case)}

{_SUBTYPE_USER_PROMPT_REMINDER}

Dieser Fall ist bereits als hämorrhagisch (klasse=1) bestätigt.
Bestimme NUR den Subtyp (historisch / nicht_akut / akut) und gib NUR das JSON-Objekt zurück.
"""


def build_messages(case: ClinicalCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": build_user_prompt(case)},
    ]


def build_binary_messages(case: ClinicalCase) -> list[dict[str, str]]:
    """Stage 1 messages: binary classification."""
    return [
        {"role": "system", "content": load_binary_system_prompt()},
        {"role": "user", "content": build_binary_user_prompt(case)},
    ]


def build_subtype_messages(case: ClinicalCase) -> list[dict[str, str]]:
    """Stage 2 messages: subtype classification (only for klasse=1)."""
    return [
        {"role": "system", "content": load_subtype_system_prompt()},
        {"role": "user", "content": build_subtype_user_prompt(case)},
    ]


def prompt_preview(case: ClinicalCase, max_chars: int = PROMPT_PREVIEW_MAX_CHARS) -> str:
    """Preview of both inference stages (binary + subtype)."""
    binary_system = load_binary_system_prompt()
    binary_user = build_binary_user_prompt(case)
    subtype_system = load_subtype_system_prompt()
    subtype_user = build_subtype_user_prompt(case)
    combined = (
        f"[STAGE 1 / SYSTEM]\n{binary_system}\n\n[STAGE 1 / USER]\n{binary_user}\n\n"
        f"[STAGE 2 / SYSTEM]\n{subtype_system}\n\n[STAGE 2 / USER]\n{subtype_user}"
    )
    if len(combined) <= max_chars:
        return combined
    return combined[: max_chars - 1] + "…"


_FALLBACK_SYSTEM_PROMPT = """Du bist ein klinisches Entscheidungssystem für zweistufige Fallklassifikation.
STUFE 1: klasse=0 nicht_hämorrhagisch vs. klasse=1 hämorrhagisch.
STUFE 2 (nur wenn klasse=1): haemorrhage_subtype ∈ {historisch, nicht_akut, akut} (PFLICHT).
Eine historische Blutung ist weiterhin eine Blutung: klasse=1, label=hämorrhagisch, subtype=historisch. NIEMALS klasse=0.
Klassifiziere NICHT als nicht_hämorrhagisch, nur weil die Blutung nicht akut oder nicht aktuell ist.
Verify_Vaskulär ist KEINE Klasse, nur Metadaten, und darf die Klassifikation nicht beeinflussen.
Bei klasse=0 ist haemorrhage_subtype=null. Gib NICHT «unbekannt» als Subtyp aus.
Antwort möglichst kompakt: max. 3 evidenz-Einträge, textstelle max. 200 Zeichen, interpretation max. 1 Satz, begruendung max. 2 kurze Sätze. Ziel < 1500 Zeichen.
Antworte ausschliesslich mit einem JSON-Objekt auf Deutsch (Feldinhalte), ohne Markdown.
"""

_USER_PROMPT_REMINDER = """Erinnerung:
- Zweistufig: erst klasse 0/1 (nicht_hämorrhagisch vs. hämorrhagisch); wenn klasse=1, dann Subtyp historisch/nicht_akut/akut (PFLICHT).
- Eine historische Blutung ist weiterhin eine Blutung → klasse=1, label=hämorrhagisch, subtype=historisch (NIEMALS klasse=0).
- Akute/frische/aktuelle Blutung → subtype=akut.
- Aktuell fallrelevante, aber nicht akute hämorrhagische Läsion → subtype=nicht_akut (NICHT nicht_hämorrhagisch).
- «geblutetes/eingeblutetes Kavernom» beschreibt ein Blutungsereignis → klasse=1; Subtyp je nach Zeitbezug. Kavernom OHNE Einblutung → nicht_hämorrhagisch.
- Verify_Vaskulär darf die Klassifikation nicht beeinflussen.
- nicht_hämorrhagisch (klasse=0) nur ohne jegliche hämorrhagische Evidenz → haemorrhage_subtype=null.
- Antwort möglichst kompakt: max. 3 evidenz-Einträge, textstelle max. 200 Zeichen, interpretation max. 1 Satz, begruendung max. 2 kurze Sätze, Ziel < 1500 Zeichen. Lange Zitate vermeiden. Nur gültiges JSON."""


# --- Two-stage hierarchical inference prompts ---------------------------------

_FALLBACK_BINARY_SYSTEM_PROMPT = """Du bist ein klinisches Entscheidungssystem für EINE Entscheidung: hämorrhagisch vs. nicht_hämorrhagisch.
Entscheide NUR klasse=0 (nicht_hämorrhagisch) oder klasse=1 (hämorrhagisch). KEIN Subtyp.
Eine historische Blutung ist weiterhin eine Blutung → klasse=1. NIEMALS klasse=0.
Klassifiziere NICHT als nicht_hämorrhagisch, nur weil die Blutung nicht akut/aktuell ist.
Verify_Vaskulär ist nur Metadaten und darf die Entscheidung nicht beeinflussen.
Antwort kompakt: max. 3 evidenz-Einträge, textstelle max. 200 Zeichen, begruendung max. 2 Sätze.
Antworte ausschliesslich mit einem JSON-Objekt auf Deutsch, ohne Markdown.
"""

_FALLBACK_SUBTYPE_SYSTEM_PROMPT = """Du bist ein klinisches Entscheidungssystem für EINE Entscheidung: den hämorrhagischen Subtyp.
Es ist bereits bestätigt, dass der Fall HÄMORRHAGISCH ist (klasse=1). Stelle das NICHT in Frage.
Wähle NUR den Subtyp: historisch, nicht_akut oder akut.
«historisch» bedeutet NICHT einfach «nicht akut». Entscheidend ist die aktuelle klinische Relevanz:
- historisch = NUR Hintergrund-Anamnese, KEINE Relevanz für aktuelle Symptome/Operation/Diagnose/Behandlung.
- nicht_akut = nicht frisch/akut, ABER aktuell klinisch relevant (z.B. eingeblutetes Kavernom verursacht aktuelle Symptome, frühere Blutung führte zur OP, Hämosiderin relevant für aktuelle Läsion).
- akut = akute/frische/subakute Blutung oder dringende akute blutungsbezogene Behandlung (z.B. Hämatomevakuation, Notfall-OP).
Entscheidungsregel: (1) nur Hintergrund-Anamnese ohne aktuelle Relevanz? ja→historisch, sonst→ (2) akute/frische/dringende Blutung? ja→akut, nein→nicht_akut.
Antwort kompakt: max. 3 evidenz-Einträge, begruendung max. 2 Sätze.
Antworte ausschliesslich mit einem JSON-Objekt auf Deutsch, ohne Markdown.
"""

_BINARY_USER_PROMPT_REMINDER = """Erinnerung (NUR Stufe 1 — binär):
- Entscheide ausschliesslich: nicht_hämorrhagisch (klasse=0) oder hämorrhagisch (klasse=1). KEINEN Subtyp wählen.
- Eine historische/vergangene Blutung ist weiterhin eine Blutung → klasse=1 (NIEMALS klasse=0).
- Klassifiziere NICHT als nicht_hämorrhagisch, nur weil die Blutung nicht akut/aktuell ist.
- «geblutetes/eingeblutetes Kavernom» beschreibt ein Blutungsereignis → klasse=1. Kavernom OHNE Einblutung → klasse=0.
- Verify_Vaskulär darf die Entscheidung nicht beeinflussen.
- Antwort kompakt, nur gültiges JSON."""

_SUBTYPE_USER_PROMPT_REMINDER = """Erinnerung (NUR Stufe 2 — Subtyp):
- Die Blutung ist bereits bestätigt. Beurteile NICHT erneut hämorrhagisch vs. nicht_hämorrhagisch.
- Wähle genau einen Subtyp: historisch, nicht_akut oder akut.
- «historisch» bedeutet NICHT einfach «nicht akut» — entscheidend ist die aktuelle klinische Relevanz.
- historisch = NUR Hintergrund-Anamnese ohne Relevanz für aktuelle Symptome/Operation/Diagnose/Behandlung.
- nicht_akut = nicht frisch/akut, ABER aktuell klinisch relevant (z.B. eingeblutetes Kavernom verursacht aktuelle Symptome, frühere Blutung führte zur OP, Hämosiderin relevant für aktuelle Läsion).
- akut = akute/frische/subakute Blutung oder dringende Behandlung (Hämatomevakuation, Notfall-OP).
- Entscheidungsregel: (1) nur Hintergrund-Anamnese ohne aktuelle Relevanz? ja→historisch, sonst→ (2) akute/frische/dringende Blutung? ja→akut, nein→nicht_akut.
- Verify_Vaskulär darf die Subtyp-Entscheidung nicht beeinflussen.
- Antwort kompakt, nur gültiges JSON."""
