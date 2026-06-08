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

PROMPT_PREVIEW_MAX_CHARS = 2000


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM_PROMPT


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


def build_user_prompt(case: ClinicalCase) -> str:
    """User message: case metadata + structured case text."""
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
---

{_USER_PROMPT_REMINDER}

Analysiere diesen klinischen Fall und gib NUR das JSON-Objekt gemäss Schema zurück.
"""


def build_messages(case: ClinicalCase) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": build_user_prompt(case)},
    ]


def prompt_preview(case: ClinicalCase, max_chars: int = PROMPT_PREVIEW_MAX_CHARS) -> str:
    system = load_system_prompt()
    user = build_user_prompt(case)
    combined = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
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
Antworte ausschliesslich mit einem JSON-Objekt auf Deutsch (Feldinhalte), ohne Markdown.
"""

_USER_PROMPT_REMINDER = """Erinnerung:
- Zweistufig: erst klasse 0/1 (nicht_hämorrhagisch vs. hämorrhagisch); wenn klasse=1, dann Subtyp historisch/nicht_akut/akut (PFLICHT).
- Eine historische Blutung ist weiterhin eine Blutung → klasse=1, label=hämorrhagisch, subtype=historisch (NIEMALS klasse=0).
- Akute/frische/aktuelle Blutung → subtype=akut.
- Aktuell fallrelevante, aber nicht akute hämorrhagische Läsion → subtype=nicht_akut (NICHT nicht_hämorrhagisch).
- «geblutetes/eingeblutetes Kavernom» beschreibt ein Blutungsereignis → klasse=1; Subtyp je nach Zeitbezug. Kavernom OHNE Einblutung → nicht_hämorrhagisch.
- Verify_Vaskulär darf die Klassifikation nicht beeinflussen.
- nicht_hämorrhagisch (klasse=0) nur ohne jegliche hämorrhagische Evidenz → haemorrhage_subtype=null."""
