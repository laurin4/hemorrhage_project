"""
Proof-of-concept demo: watch the LLM extraction work, step by step, on ONE case.

Makes the pipeline transparent for a presentation. For a single clinical case it
prints, in order:

  1. Case identity + which reports are present
  2. The structured clinical free-text the model actually sees
  3. STAGE 1 (binary) — the exact system + user prompt we send
  4. STAGE 1 — the raw JSON the LLM returns
  5. STAGE 1 — the parsed, structured result (klasse / label / confidence / reason)
  6. STAGE 2 (subtype) — same prompt → raw → parsed (only if Stage 1 = hämorrhagisch)
  7. FINAL — the resulting class and which spreadsheet one-hot column it fills

Sources for the LLM responses (slowest → fastest):
  --live              call the real LLM now (needs the USZ/Ollama server up; slow)
  --replay            reuse raw responses already stored in the predictions CSV
                      (needs the data files, but no LLM)
  --from-snapshot P   replay a frozen, self-contained JSON snapshot — NO LLM and
                      NO data files. Runs instantly anywhere (meeting room / laptop).

Prepare a frozen snapshot once (where data + predictions exist), then present from it:
  python3 -m src.tasks.hemorrhage.demo_extraction --snapshot          # writes the JSON
  python3 -m src.tasks.hemorrhage.demo_extraction --from-snapshot ... # instant replay

Examples
--------
  python3 -m src.tasks.hemorrhage.demo_extraction                       # live, auto-pick
  python3 -m src.tasks.hemorrhage.demo_extraction --case-id CASE        # specific case
  python3 -m src.tasks.hemorrhage.demo_extraction --replay              # offline-ish replay
  python3 -m src.tasks.hemorrhage.demo_extraction --snapshot            # freeze one case
  python3 -m src.tasks.hemorrhage.demo_extraction --from-snapshot data/demo/demo_extraction_case.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.case.models import ClinicalCase
from src.pipeline.paths import DATA_DIR, HEMORRHAGE_CASE_PREDICTIONS_PATH
from src.tasks.hemorrhage.export.classification_merge import (
    STATUS_OK,
    classify_prediction,
)
from src.tasks.hemorrhage.inference.parse import (
    parse_binary_response,
    parse_subtype_response,
)
from src.tasks.hemorrhage.inference.prompt import (
    build_binary_messages,
    build_subtype_messages,
)
from src.tasks.hemorrhage.io.load_cases import load_clinical_cases

SUBTYPE_STAGE_DELIMITER = "--- SUBTYPE STAGE ---"
DEFAULT_SNAPSHOT_PATH = DATA_DIR / "demo" / "demo_extraction_case.json"

BLEEDING_HINTS = (
    "blutung",
    "einblut",
    "eingeblut",
    "geblutet",
    "hämatom",
    "haematom",
    "hämorrhag",
    "haemorrhag",
    "hemorrhag",
)

# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def _banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def _step(n: str, title: str) -> None:
    print(f"\n{'-' * 78}\nSCHRITT {n}: {title}\n{'-' * 78}")


def _note(text: str) -> None:
    print(f"  → {text}")


def _block(text: str, *, max_chars: Optional[int] = None) -> None:
    body = "" if text is None else str(text)
    if max_chars and len(body) > max_chars:
        body = body[:max_chars] + f"\n… [gekürzt, {len(str(text))} Zeichen gesamt]"
    for ln in body.splitlines() or [""]:
        print(f"    {ln}")


# --------------------------------------------------------------------------- #
# Case selection
# --------------------------------------------------------------------------- #
def _case_has_bleeding_hint(case: ClinicalCase) -> bool:
    text = case.structured_case_text().lower()
    return any(h in text for h in BLEEDING_HINTS)


def _first_case_id(preds: pd.DataFrame, mask: "pd.Series") -> Optional[str]:
    hits = preds[mask]
    if hits.empty:
        return None
    return str(hits.iloc[0].get("case_id", "")).strip() or None


def _autopick_from_predictions(
    preds: pd.DataFrame,
    exclude_pids: Optional[frozenset[str]] = None,
) -> Optional[str]:
    """
    Pick a *correct* hämorrhagisch case (true positive) with a subtype, so both
    stages show and the demo never displays a misclassification. Falls back to any
    hämorrhagisch+subtype case if the reference label is unavailable. Patients in
    ``exclude_pids`` are never selected.
    """
    if preds.empty or "label" not in preds.columns:
        return None
    if exclude_pids and "excel_pid" in preds.columns:
        norm = {str(p).strip() for p in exclude_pids}
        keep = ~preds["excel_pid"].astype(str).str.strip().isin(norm)
    else:
        keep = pd.Series(True, index=preds.index)
    base = (
        keep
        & preds.get("status", "").astype(str).str.strip().eq("success")
        & preds["label"].astype(str).str.strip().str.lower().eq("hämorrhagisch")
        & preds.get("haemorrhage_subtype", "").astype(str).str.strip().ne("")
    )
    if "reference_label_status" in preds.columns:
        correct = base & preds["reference_label_status"].astype(str).str.strip().str.lower().eq(
            "hemorrhagic"
        )
        picked = _first_case_id(preds, correct)
        if picked:
            return picked
    return _first_case_id(preds, base)


def _select_case(
    cases: List[ClinicalCase],
    *,
    case_id: Optional[str],
    preds: Optional[pd.DataFrame],
) -> Optional[ClinicalCase]:
    by_id = {c.case_id: c for c in cases}
    if case_id:
        return by_id.get(case_id)
    if preds is not None:
        picked = _autopick_from_predictions(preds)
        if picked and picked in by_id:
            return by_id[picked]
    for c in cases:
        if _case_has_bleeding_hint(c):
            return c
    return cases[0] if cases else None


# --------------------------------------------------------------------------- #
# Raw response sourcing (live vs replay)
# --------------------------------------------------------------------------- #
def _replay_raw_for_case(preds: pd.DataFrame, case_id: str) -> str:
    row = preds[preds.get("case_id", "").astype(str) == case_id]
    if row.empty:
        return ""
    return str(row.iloc[0].get("raw_llm_response", "") or "")


def _split_replay(raw_combined: str) -> tuple[str, str]:
    """Split a stored raw response into (binary_raw, subtype_raw)."""
    if SUBTYPE_STAGE_DELIMITER in raw_combined:
        binary_raw, subtype_raw = raw_combined.split(SUBTYPE_STAGE_DELIMITER, 1)
        return binary_raw.strip(), subtype_raw.strip()
    return raw_combined.strip(), ""


# --------------------------------------------------------------------------- #
# Trace: a self-contained record of one case's full extraction (snapshot-ready)
# --------------------------------------------------------------------------- #
def build_trace(
    case: ClinicalCase,
    *,
    binary_raw: str,
    subtype_raw: str,
    stage2_ran: bool,
    mode: str,
    reports_file: str = "",
) -> Dict[str, Any]:
    """Assemble the full step-by-step trace dict from a case + raw LLM responses."""
    binary_messages = build_binary_messages(case)
    binary_result = parse_binary_response(binary_raw, context=f"demo_binary:{case.case_id}")
    pred = binary_result.prediction

    trace: Dict[str, Any] = {
        "mode": mode,
        "reports_file": reports_file,
        "case": {
            "case_id": case.case_id,
            "excel_pid": case.excel_pid,
            "excel_opdat": case.excel_opdat,
            "opber_fallnr": case.opber_fallnr,
            "available_report_types": list(case.available_report_types),
        },
        "input_text": case.structured_case_text(),
        "reports": [
            {
                "typus_label": case.reports[code].typus_label,
                "report_text": case.reports[code].report_text,
            }
            for code in sorted(case.reports.keys())
        ],
        "stage1": {
            "system_prompt": binary_messages[0]["content"],
            "user_prompt": binary_messages[1]["content"],
            "raw_response": binary_raw,
            "success": binary_result.success,
            "error": binary_result.error_message,
            "parsed": {
                "klasse": pred.get("klasse"),
                "label": pred.get("label"),
                "sicherheit": pred.get("sicherheit"),
                "begruendung": pred.get("begruendung"),
                "evidenz": pred.get("evidenz") or [],
            },
        },
        "stage2": None,
    }

    label = pred.get("label")
    subtype: Optional[str] = None

    if binary_result.success and pred.get("klasse") == 1:
        subtype_messages = build_subtype_messages(case)
        subtype_result = parse_subtype_response(
            subtype_raw, context=f"demo_subtype:{case.case_id}"
        )
        subtype = subtype_result.haemorrhage_subtype
        trace["stage2"] = {
            "ran": stage2_ran,
            "system_prompt": subtype_messages[0]["content"],
            "user_prompt": subtype_messages[1]["content"],
            "raw_response": subtype_raw,
            "success": subtype_result.success,
            "parsed": {
                "haemorrhage_subtype": subtype_result.haemorrhage_subtype,
                "sicherheit": subtype_result.sicherheit,
                "begruendung": subtype_result.begruendung,
                "evidenz": subtype_result.evidenz or [],
            },
        }

    classification = classify_prediction(label, subtype, "success")
    trace["final"] = {
        "label": label,
        "subtype": subtype,
        "class_column": classification.class_column,
        "status": classification.status,
    }
    return trace


def render_trace(trace: Dict[str, Any], *, max_text_chars: Optional[int] = 1500) -> None:
    """Print the 7-step narration purely from a trace dict (no data/LLM needed)."""
    case = trace.get("case", {})
    stage1 = trace.get("stage1", {})
    stage2 = trace.get("stage2")
    final = trace.get("final", {})

    _banner("DEMO — LLM-Extraktion aus klinischem Freitext (Proof of Concept)")
    print(f"Modus: {trace.get('mode', '')}")
    if trace.get("reports_file"):
        print(f"Reports-Datei: {trace['reports_file']}")

    _step("1", "Welcher Fall? (1 Fall = 1 Operation, mehrere Berichte)")
    _note("Ein Fall bündelt Operations-, Eintritts- und Austrittsbericht.")
    print(f"    case_id      = {case.get('case_id')}")
    print(f"    excel_pid    = {case.get('excel_pid')}")
    print(f"    excel_opdat  = {case.get('excel_opdat')}")
    print(f"    opber_fallnr = {case.get('opber_fallnr')}")
    reports = ", ".join(case.get("available_report_types", [])) or "(keine)"
    print(f"    Berichte vorhanden = {reports}")

    _step("2", "Der unstrukturierte klinische Freitext (Eingabe ins Modell)")
    _note("Genau dieser Text — ohne Vorverarbeitung/NLP — wird dem LLM übergeben.")
    _block(trace.get("input_text") or "(kein Berichtstext)", max_chars=max_text_chars)

    _step("3", "STUFE 1 (binär) — der exakte Prompt an das LLM")
    _note("Frage an das Modell: hämorrhagisch (klasse=1) oder nicht (klasse=0)?")
    print("\n  [SYSTEM-PROMPT]")
    _block(stage1.get("system_prompt", ""), max_chars=max_text_chars)
    print("\n  [USER-PROMPT]")
    _block(stage1.get("user_prompt", ""), max_chars=max_text_chars)

    _step("4", "STUFE 1 — die ROHE Antwort des LLM (JSON)")
    _block(stage1.get("raw_response", ""))

    _step("5", "STUFE 1 — geparste, strukturierte Ausgabe")
    _note("Aus dem freien JSON werden geprüfte, kanonische Felder.")
    if not stage1.get("success"):
        print(f"    parse_failed: {stage1.get('error')}")
    p1 = stage1.get("parsed", {})
    print(f"    klasse        = {p1.get('klasse')}")
    print(f"    label         = {p1.get('label')}")
    print(f"    sicherheit    = {p1.get('sicherheit')}")
    print(f"    kurzbegründung= {p1.get('begruendung')}")

    if stage2 is not None:
        _step("6", "STUFE 2 (Subtyp) — läuft NUR, weil Stufe 1 = hämorrhagisch")
        _note("Frage: historisch / nicht_akut / akut? (Blutung gilt als bestätigt)")
        print("\n  [SYSTEM-PROMPT]")
        _block(stage2.get("system_prompt", ""), max_chars=max_text_chars)
        print("\n  [USER-PROMPT]")
        _block(stage2.get("user_prompt", ""), max_chars=max_text_chars)
        _step("6b", "STUFE 2 — ROHE Antwort + geparster Subtyp")
        print("  [ROH]")
        _block(stage2.get("raw_response") or "(keine Antwort)")
        p2 = stage2.get("parsed", {})
        print("\n  [GEPARST]")
        print(f"    haemorrhage_subtype = {p2.get('haemorrhage_subtype')}")
        print(f"    sicherheit          = {p2.get('sicherheit')}")
        print(f"    begründung          = {p2.get('begruendung')}")
    else:
        _step("6", "STUFE 2 übersprungen")
        _note("Stufe 1 = nicht_hämorrhagisch → kein Subtyp nötig (spart Zeit/Tokens).")

    _step("7", "ERGEBNIS — finale Klasse und Spalte in der Tabelle")
    print(f"    label   = {final.get('label')}")
    subtype = final.get("subtype")
    print(f"    subtyp  = {subtype if subtype is not None else '(keiner)'}")
    if final.get("class_column") and final.get("status") == STATUS_OK:
        print(f"\n    → One-hot-Spalte erhält eine 1:  «{final['class_column']}»")
        print("      (alle anderen Klassenspalten = 0)")
    else:
        print(f"\n    → Keine One-hot-Markierung. klassifikation_status = «{final.get('status')}»")

    _banner("DEMO ENDE")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def obtain_raws(
    case: ClinicalCase,
    *,
    replay: bool,
    preds: Optional[pd.DataFrame],
) -> Optional[tuple[str, str, bool, str]]:
    """
    Source the Stage 1 / Stage 2 raw LLM responses for one case.

    Returns ``(binary_raw, subtype_raw, stage2_ran, mode)`` or ``None`` on failure.
    In replay mode the responses come from the stored predictions CSV; in live
    mode the real LLM is called (Stage 2 only if Stage 1 = hämorrhagisch).
    """
    if replay:
        if preds is None:
            print("Kein predictions CSV für Replay gefunden.")
            return None
        binary_raw, subtype_raw = _split_replay(_replay_raw_for_case(preds, case.case_id))
        if not binary_raw:
            print(
                f"Keine gespeicherte LLM-Antwort für {case.case_id} im CSV. "
                "Bitte zuerst die Pipeline laufen lassen oder --live nutzen."
            )
            return None
        return binary_raw, subtype_raw, bool(subtype_raw), "REPLAY (gespeicherte LLM-Antworten)"

    from src.tasks.hemorrhage.inference.llm_client import call_llm

    try:
        binary_raw = call_llm(build_binary_messages(case)) or ""
    except Exception as exc:  # noqa: BLE001 — demo surfaces the real error
        print(f"LLM-Aufruf (Stufe 1) fehlgeschlagen: {type(exc).__name__}: {exc}")
        return None

    # Only call Stage 2 if Stage 1 says hämorrhagisch.
    binary_result = parse_binary_response(binary_raw, context=f"demo_binary:{case.case_id}")
    subtype_raw = ""
    stage2_ran = False
    if binary_result.success and binary_result.prediction.get("klasse") == 1:
        try:
            subtype_raw = call_llm(build_subtype_messages(case)) or ""
            stage2_ran = True
        except Exception as exc:  # noqa: BLE001
            print(f"LLM-Aufruf (Subtyp) fehlgeschlagen: {type(exc).__name__}: {exc}")
    return binary_raw, subtype_raw, stage2_ran, "LIVE (echter LLM-Aufruf)"


def _build_live_or_replay_trace(
    *,
    case_id: Optional[str],
    replay: bool,
    predictions_path: Optional[Path],
    reports_path: Optional[Path],
    reference_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Run live (LLM) or replay (predictions CSV) for one case and build its trace."""
    pred_path = predictions_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    preds: Optional[pd.DataFrame] = None
    if pred_path.exists():
        preds = pd.read_csv(pred_path, dtype=str).fillna("")

    cases, _stats, reports_file, errors = load_clinical_cases(reports_path)
    if not cases:
        print("Keine Fälle geladen.")
        for e in errors:
            print(f"  - {e}")
        return None

    case = _select_case(cases, case_id=case_id, preds=preds)
    if case is None:
        print(f"Fall nicht gefunden: {case_id!r}")
        return None

    raws = obtain_raws(case, replay=replay, preds=preds)
    if raws is None:
        return None
    binary_raw, subtype_raw, stage2_ran, mode = raws

    return build_trace(
        case,
        binary_raw=binary_raw,
        subtype_raw=subtype_raw,
        stage2_ran=stage2_ran,
        mode=mode,
        reports_file=str(reports_file),
    )


def run_demo(
    *,
    case_id: Optional[str] = None,
    replay: bool = False,
    from_snapshot: Optional[Path] = None,
    snapshot_out: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    reports_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    max_text_chars: Optional[int] = 1500,
) -> int:
    # 1. Instant offline replay from a frozen snapshot (no data, no LLM).
    if from_snapshot is not None:
        if not from_snapshot.exists():
            print(f"Snapshot nicht gefunden: {from_snapshot}")
            return 1
        trace = json.loads(from_snapshot.read_text(encoding="utf-8"))
        trace["mode"] = "SNAPSHOT (vorbereiteter Fall — kein LLM, keine Daten nötig)"
        render_trace(trace, max_text_chars=max_text_chars)
        return 0

    # 2. Build a fresh trace via live LLM or replay from predictions CSV.
    trace = _build_live_or_replay_trace(
        case_id=case_id,
        replay=replay,
        predictions_path=predictions_path,
        reports_path=reports_path,
        reference_path=reference_path,
    )
    if trace is None:
        return 1

    # 3. Optionally freeze it to a self-contained JSON for offline presentation.
    if snapshot_out is not None:
        snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        snapshot_out.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Snapshot gespeichert: {snapshot_out}")
        print("Damit jederzeit offline abspielbar:")
        print(f"  python3 -m src.tasks.hemorrhage.demo_extraction --from-snapshot {snapshot_out}")
        return 0

    render_trace(trace, max_text_chars=max_text_chars)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step-by-step demo of the LLM hemorrhage extraction on one case."
    )
    parser.add_argument("--case-id", type=str, default=None, help="Specific case_id to demo")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Reuse stored LLM responses from the predictions CSV (no LLM call)",
    )
    parser.add_argument("--live", action="store_true", help="Force live LLM call (default)")
    parser.add_argument(
        "--snapshot",
        nargs="?",
        const=str(DEFAULT_SNAPSHOT_PATH),
        default=None,
        help="Freeze the selected case to a self-contained JSON (default path if no value).",
    )
    parser.add_argument(
        "--from-snapshot",
        type=Path,
        default=None,
        help="Replay a frozen JSON snapshot instantly (no LLM, no data files).",
    )
    parser.add_argument("--predictions", type=Path, default=None, help="Predictions CSV (replay source)")
    parser.add_argument("--reports", type=Path, default=None, help="Reports Excel input")
    parser.add_argument("--reference", type=Path, default=None, help="Reference Excel (optional)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full text/prompts (default truncates long blocks for readability)",
    )
    args = parser.parse_args(argv)

    # Building a snapshot defaults to replay (fast) unless --live is explicitly set.
    snapshot_out = Path(args.snapshot) if args.snapshot is not None else None
    replay = args.replay or (snapshot_out is not None and not args.live)

    return run_demo(
        case_id=args.case_id,
        replay=replay and not args.live,
        from_snapshot=args.from_snapshot,
        snapshot_out=snapshot_out,
        predictions_path=args.predictions,
        reports_path=args.reports,
        reference_path=args.reference,
        max_text_chars=None if args.full else 1500,
    )


if __name__ == "__main__":
    sys.exit(main())
