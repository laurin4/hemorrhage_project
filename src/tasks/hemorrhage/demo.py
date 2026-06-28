"""
Polished, interactive walkthrough of the prompt-based extraction pipeline.

This demo is a *blueprint*: it shows how an LLM + prompt-engineering pipeline turns
an unstructured German clinical report into validated, structured information. The
hemorrhage task is just the worked example — the same five stages (reports →
prompt engineering → LLM → JSON validation → structured output → spreadsheet)
apply to any medical extraction task.

It runs INSTANTLY and NEVER calls the LLM during a presentation: it replays
previously captured real responses stored in self-contained JSON snapshots under
``data/demo/``.

Run the demo:
    python3 -m src.tasks.hemorrhage.demo                # interactive menu
    python3 -m src.tasks.hemorrhage.demo --positive     # positive case only
    python3 -m src.tasks.hemorrhage.demo --negative     # negative case only
    python3 -m src.tasks.hemorrhage.demo --both         # both, back to back

(Re)generate the demo snapshots (where data + predictions exist, e.g. the server):
    python3 -m src.tasks.hemorrhage.demo --snapshot-positive
    python3 -m src.tasks.hemorrhage.demo --snapshot-negative
    python3 -m src.tasks.hemorrhage.demo --snapshot-positive --case-id <case_id>

See docs/demo/DEMO_GUIDE.md for a full guide.
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
from src.tasks.hemorrhage.demo_extraction import (
    _autopick_from_predictions,
    build_trace,
    obtain_raws,
)
from src.tasks.hemorrhage.io.load_cases import load_clinical_cases

DEMO_DIR = DATA_DIR / "demo"
POSITIVE_SNAPSHOT = DEMO_DIR / "positive_case.json"
NEGATIVE_SNAPSHOT = DEMO_DIR / "negative_case.json"

SEP = "=" * 60
THIN = "-" * 60
RULE = "\u2014" * 36  # em-dash rule for the final box

SYSTEM_PROMPT_EXCERPT_CHARS = 1100
TEXT_BLOCK_CHARS = 1400

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
# Formatting helpers
# --------------------------------------------------------------------------- #
def _pause(enabled: bool, last: bool = False) -> None:
    if not enabled:
        return
    label = "Press ENTER to finish…" if last else "Press ENTER to continue…"
    try:
        input(f"\n{label} ")
    except EOFError:
        pass


def _step(n: int, title: str) -> None:
    print(f"\n{SEP}")
    print(f"STEP {n}  ·  {title}")
    print(SEP)


def _explain(text: str) -> None:
    print(f"\n{text}\n")


def _block(text: object, *, limit: Optional[int] = None, indent: str = "    ") -> None:
    body = "" if text is None else str(text)
    if limit and len(body) > limit:
        body = body[:limit] + f"\n… [truncated · {len(str(text))} chars total]"
    for ln in body.splitlines() or [""]:
        print(f"{indent}{ln}")


def _excerpt(text: str, limit: Optional[int]) -> str:
    s = str(text or "")
    if limit and len(s) > limit:
        return s[:limit] + f"\n… [truncated · full prompt has {len(s)} chars]"
    return s


def _collapse_evidence(user_prompt: str, input_text: str) -> str:
    """Replace the (already-shown) clinical text inside the user prompt with a marker."""
    if input_text and input_text in user_prompt:
        return user_prompt.replace(input_text, "‹[clinical text — shown in STEP 2]›")
    return user_prompt


def _is_hemorrhagic(stage1: Dict[str, Any]) -> bool:
    parsed = stage1.get("parsed", {})
    if parsed.get("klasse") == 1:
        return True
    return str(parsed.get("label", "")).strip().lower() == "hämorrhagisch"


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
def present_case(trace: Dict[str, Any], *, pause: bool = True, full: bool = False) -> None:
    """Render one case as a clean, paced, step-by-step walkthrough."""
    text_limit = None if full else TEXT_BLOCK_CHARS
    sys_limit = None if full else SYSTEM_PROMPT_EXCERPT_CHARS

    case = trace.get("case", {})
    stage1 = trace.get("stage1", {})
    stage2 = trace.get("stage2")
    final = trace.get("final", {})
    input_text = trace.get("input_text", "")
    reports: List[Dict[str, str]] = trace.get("reports") or []

    hemorrhagic = _is_hemorrhagic(stage1)
    polarity = "POSITIVE · hämorrhagisch" if hemorrhagic else "NEGATIVE · nicht_hämorrhagisch"

    print(f"\n{SEP}")
    print(f"  CASE  {case.get('case_id', '')}")
    print(f"  {polarity}")
    print(SEP)

    # ---- STEP 1: original reports ---------------------------------------- #
    _step(1, "Original clinical reports")
    _explain("The pipeline receives completely unstructured German clinical documentation.")
    if reports:
        for rep in reports:
            print(f"  [{rep.get('typus_label', '')}]")
            _block(rep.get("report_text", ""), limit=text_limit)
            print("")
    else:
        _block(input_text or "(no report text)", limit=text_limit)
    _pause(pause)

    # ---- STEP 2: evidence block ------------------------------------------ #
    _step(2, "Evidence presented to the LLM")
    _explain(
        "The relevant report text is concatenated and forwarded as-is — no manual\n"
        "feature engineering and no NLP preprocessing. The model reads the raw text."
    )
    _block(input_text or "(no report text)", limit=text_limit)
    _pause(pause)

    # ---- STEP 3: Stage 1 prompt ------------------------------------------ #
    _step(3, "Stage 1 prompt — the binary decision")
    _explain("Engineered rules turn a vague clinical question into a deterministic instruction.")
    print("  [SYSTEM PROMPT — excerpt]")
    _block(_excerpt(stage1.get("system_prompt", ""), sys_limit))
    print("\n  [USER PROMPT]")
    _block(_collapse_evidence(stage1.get("user_prompt", ""), input_text), limit=text_limit)
    _pause(pause)

    # ---- STEP 4: raw response + parsed ----------------------------------- #
    _step(4, "Real LLM response  →  validated JSON")
    print("  [RAW LLM RESPONSE]")
    _block(stage1.get("raw_response", ""))
    print("\n  [PARSED & VALIDATED]")
    p1 = stage1.get("parsed", {})
    print(f"    klasse        = {p1.get('klasse')}")
    print(f"    label         = {p1.get('label')}")
    print(f"    sicherheit    = {p1.get('sicherheit')}")
    print(f"    begründung    = {p1.get('begruendung')}")
    _explain("The parser validates and normalises the model output before anything uses it.")
    _pause(pause)

    confidence = p1.get("sicherheit")

    # ---- Branch: Stage 2 conditional ------------------------------------- #
    if hemorrhagic and stage2 is not None:
        _step(5, "Stage 2 prompt — the subtype")
        _explain("A second, narrower prompt runs ONLY when Stage 1 found a hemorrhage.")
        print("  [SYSTEM PROMPT — excerpt]")
        _block(_excerpt(stage2.get("system_prompt", ""), sys_limit))
        print("\n  [USER PROMPT]")
        _block(_collapse_evidence(stage2.get("user_prompt", ""), input_text), limit=text_limit)
        _pause(pause)

        _step(6, "Real subtype response  →  validated JSON")
        print("  [RAW LLM RESPONSE]")
        _block(stage2.get("raw_response", "") or "(no response)")
        print("\n  [PARSED & VALIDATED]")
        p2 = stage2.get("parsed", {})
        print(f"    haemorrhage_subtype = {p2.get('haemorrhage_subtype')}")
        print(f"    sicherheit          = {p2.get('sicherheit')}")
        print(f"    begründung          = {p2.get('begruendung')}")
        confidence = p2.get("sicherheit") or confidence
        _pause(pause)
    else:
        print(f"\n{SEP}")
        print("STAGE 2 SKIPPED")
        print(SEP)
        _explain(
            "Reason: No hemorrhage detected — the subtype question does not apply.\n"
            "This is the efficiency of the hierarchical pipeline: the second, more\n"
            "expensive LLM call is avoided entirely."
        )
        _pause(pause)

    # ---- STEP 7: final output + pipeline summary ------------------------- #
    _step(7, "Final structured output")
    subtype = final.get("subtype")
    class_column = final.get("class_column")
    print(f"  {RULE}")
    print("  Final Classification")
    print("")
    print(f"  Hemorrhagic:         {'YES' if hemorrhagic else 'NO'}")
    print(f"  Subtype:             {subtype if subtype else '—'}")
    print(f"  Confidence:          {confidence or '—'}")
    print(f"  Spreadsheet column:  {class_column if class_column else '— (no one-hot marker)'}")
    print(f"  {RULE}")

    _explain("Pipeline summary — the reusable blueprint:")
    for i, stage in enumerate(
        [
            "Clinical reports (unstructured German free-text)",
            "Prompt Engineering (rules + schema)",
            "LLM (local model)",
            "JSON Validation (parser + normalisation)",
            "Structured Output (klasse / subtype / confidence)",
            "Spreadsheet (one-hot columns)",
        ]
    ):
        print(f"      {stage}")
        if i < 5:
            print("                       ↓")
    _pause(pause, last=True)


# --------------------------------------------------------------------------- #
# Snapshot generation
# --------------------------------------------------------------------------- #
def _has_bleeding_hint(case: ClinicalCase) -> bool:
    text = case.structured_case_text().lower()
    return any(h in text for h in BLEEDING_HINTS)


def _autopick_negative_from_predictions(preds: pd.DataFrame) -> Optional[str]:
    """Pick a case predicted nicht_hämorrhagisch (Stage 2 will be skipped)."""
    if preds.empty or "label" not in preds.columns:
        return None
    mask = preds.get("status", "").astype(str).str.strip().eq("success") & preds[
        "label"
    ].astype(str).str.strip().str.lower().eq("nicht_hämorrhagisch")
    hits = preds[mask]
    if hits.empty:
        return None
    return str(hits.iloc[0].get("case_id", "")).strip() or None


def _select_polarity_case(
    cases: List[ClinicalCase],
    preds: Optional[pd.DataFrame],
    *,
    kind: str,
    case_id: Optional[str],
) -> Optional[ClinicalCase]:
    by_id = {c.case_id: c for c in cases}
    if case_id:
        return by_id.get(case_id)

    if preds is not None:
        picked = (
            _autopick_from_predictions(preds)
            if kind == "positive"
            else _autopick_negative_from_predictions(preds)
        )
        if picked and picked in by_id:
            return by_id[picked]

    # Fallback heuristic when no predictions are available.
    want_bleeding = kind == "positive"
    for c in cases:
        if _has_bleeding_hint(c) == want_bleeding:
            return c
    return cases[0] if cases else None


def generate_snapshot(
    *,
    kind: str,
    out_path: Path,
    case_id: Optional[str] = None,
    live: bool = False,
    predictions_path: Optional[Path] = None,
    reports_path: Optional[Path] = None,
) -> int:
    """Freeze one case (positive/negative) to a self-contained JSON snapshot."""
    pred_path = predictions_path or HEMORRHAGE_CASE_PREDICTIONS_PATH
    preds: Optional[pd.DataFrame] = None
    if pred_path.exists():
        preds = pd.read_csv(pred_path, dtype=str).fillna("")

    cases, _stats, reports_file, errors = load_clinical_cases(reports_path)
    if not cases:
        print("No cases loaded — cannot build a snapshot.")
        for e in errors:
            print(f"  - {e}")
        return 1

    case = _select_polarity_case(cases, preds, kind=kind, case_id=case_id)
    if case is None:
        print(f"Could not find a {kind} case (case_id={case_id!r}).")
        return 1

    replay = (not live) and (preds is not None)
    raws = obtain_raws(case, replay=replay, preds=preds)
    if raws is None:
        if not live:
            print("Tip: pass --live to call the LLM, or run the pipeline first to capture responses.")
        return 1
    binary_raw, subtype_raw, stage2_ran, mode = raws

    trace = build_trace(
        case,
        binary_raw=binary_raw,
        subtype_raw=subtype_raw,
        stage2_ran=stage2_ran,
        mode=mode,
        reports_file=str(reports_file),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    label = trace["final"].get("label")
    print(f"Saved {kind} snapshot → {out_path}")
    print(f"  case_id = {case.case_id}   label = {label}   subtype = {trace['final'].get('subtype')}")
    return 0


# --------------------------------------------------------------------------- #
# Loading + interactive menu
# --------------------------------------------------------------------------- #
def load_trace(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _missing_snapshot_hint(which: str, path: Path, flag: str) -> None:
    print(f"\n[!] {which} snapshot not found: {path}")
    print(f"    Generate it (where data + predictions exist):")
    print(f"      python3 -m src.tasks.hemorrhage.demo {flag}")


def interactive_menu(*, pause: bool = True, full: bool = False) -> int:
    positive = load_trace(POSITIVE_SNAPSHOT)
    negative = load_trace(NEGATIVE_SNAPSHOT)

    while True:
        print(f"\n{SEP}")
        print("  HEMORRHAGE DEMO  ·  LLM extraction pipeline")
        print(SEP)
        print("  Choose demonstration:\n")
        print("    [1]  Positive hemorrhagic case")
        print("    [2]  Negative non-hemorrhagic case")
        print("    [3]  Run both")
        print("    [q]  Quit")
        try:
            choice = input("\n  > ").strip().lower()
        except EOFError:
            return 0

        if choice in ("q", "quit", "exit"):
            print("Bye.")
            return 0
        if choice == "1":
            if positive is None:
                _missing_snapshot_hint("Positive", POSITIVE_SNAPSHOT, "--snapshot-positive")
            else:
                present_case(positive, pause=pause, full=full)
        elif choice == "2":
            if negative is None:
                _missing_snapshot_hint("Negative", NEGATIVE_SNAPSHOT, "--snapshot-negative")
            else:
                present_case(negative, pause=pause, full=full)
        elif choice == "3":
            if positive is None:
                _missing_snapshot_hint("Positive", POSITIVE_SNAPSHOT, "--snapshot-positive")
            else:
                present_case(positive, pause=pause, full=full)
            if negative is None:
                _missing_snapshot_hint("Negative", NEGATIVE_SNAPSHOT, "--snapshot-negative")
            else:
                present_case(negative, pause=pause, full=full)
        else:
            print("  Please choose 1, 2, 3 or q.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Interactive, instant demo of the LLM extraction pipeline (no live LLM)."
    )
    parser.add_argument("--positive", action="store_true", help="Present the positive case only")
    parser.add_argument("--negative", action="store_true", help="Present the negative case only")
    parser.add_argument("--both", action="store_true", help="Present both cases back to back")
    parser.add_argument("--no-pause", action="store_true", help="Do not pause between sections")
    parser.add_argument("--full", action="store_true", help="Show full prompts / text (no truncation)")

    parser.add_argument(
        "--snapshot-positive", action="store_true", help="(Re)generate data/demo/positive_case.json"
    )
    parser.add_argument(
        "--snapshot-negative", action="store_true", help="(Re)generate data/demo/negative_case.json"
    )
    parser.add_argument("--case-id", type=str, default=None, help="Pick a specific case for snapshot")
    parser.add_argument("--live", action="store_true", help="Call the real LLM when generating a snapshot")
    parser.add_argument("--predictions", type=Path, default=None, help="Predictions CSV (replay source)")
    parser.add_argument("--reports", type=Path, default=None, help="Reports Excel input")
    args = parser.parse_args(argv)

    if args.snapshot_positive:
        return generate_snapshot(
            kind="positive",
            out_path=POSITIVE_SNAPSHOT,
            case_id=args.case_id,
            live=args.live,
            predictions_path=args.predictions,
            reports_path=args.reports,
        )
    if args.snapshot_negative:
        return generate_snapshot(
            kind="negative",
            out_path=NEGATIVE_SNAPSHOT,
            case_id=args.case_id,
            live=args.live,
            predictions_path=args.predictions,
            reports_path=args.reports,
        )

    pause = not args.no_pause
    if args.positive or args.negative or args.both:
        if args.positive or args.both:
            trace = load_trace(POSITIVE_SNAPSHOT)
            if trace is None:
                _missing_snapshot_hint("Positive", POSITIVE_SNAPSHOT, "--snapshot-positive")
            else:
                present_case(trace, pause=pause, full=args.full)
        if args.negative or args.both:
            trace = load_trace(NEGATIVE_SNAPSHOT)
            if trace is None:
                _missing_snapshot_hint("Negative", NEGATIVE_SNAPSHOT, "--snapshot-negative")
            else:
                present_case(trace, pause=pause, full=args.full)
        return 0

    return interactive_menu(pause=pause, full=args.full)


if __name__ == "__main__":
    sys.exit(main())
