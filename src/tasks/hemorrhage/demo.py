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

# Patients (excel_pid) to NEVER auto-pick for the demo, e.g. cases whose reference
# label is clinically unreliable (the model "agrees" with a wrong label, so it would
# look like a correct true negative while actually being a missed bleeding).
DEMO_EXCLUDED_PIDS: frozenset[str] = frozenset({"10206120"})

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


def _split_report_sections(text: str) -> List[tuple[str, str]]:
    """Split stitched report text into labelled sections like [Diagnosen]."""
    sections: List[tuple[str, str]] = []
    heading = ""
    lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if heading or lines:
                body = "\n".join(lines).strip()
                if heading or body:
                    sections.append((heading or "Text", body))
            heading = stripped.strip("[]")
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if heading or body:
        sections.append((heading or "Text", body))
    if not sections and (text or "").strip():
        sections.append(("Text", text.strip()))
    return sections


def _evidenz_from_stage(stage: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return normalised evidenz items from parsed trace or raw JSON (old snapshots)."""
    parsed = stage.get("parsed") or {}
    items = parsed.get("evidenz")
    if isinstance(items, list) and items:
        return [e for e in items if isinstance(e, dict)]
    raw = str(stage.get("raw_response", "") or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(obj, dict):
        return []
    ev = obj.get("evidenz")
    if isinstance(ev, list):
        return [e for e in ev if isinstance(e, dict)]
    return []


def _render_llm_input_evidence(
    reports: List[Dict[str, str]],
    input_text: str,
    *,
    limit: Optional[int] = None,
) -> None:
    """Step 2 — show what the model reads, report-by-report with section labels."""
    if reports:
        total_chars = sum(len(str(r.get("report_text", ""))) for r in reports)
        print(
            f"  {len(reports)} report(s) · {total_chars:,} characters forwarded to the model\n"
        )
        for i, rep in enumerate(reports, start=1):
            label = rep.get("typus_label", "Report")
            body = str(rep.get("report_text", "") or "")
            print(f"  {THIN}")
            print(f"  Report {i} · {label}  ({len(body):,} chars)")
            print(f"  {THIN}")
            sections = _split_report_sections(body)
            if len(sections) > 1 or (sections and sections[0][0] != "Text"):
                for heading, section_body in sections:
                    if not section_body.strip():
                        continue
                    print(f"\n    [{heading}]")
                    _block(section_body, limit=limit, indent="      ")
            else:
                _block(body, limit=limit)
            print("")
    else:
        _block(input_text or "(no report text)", limit=limit)


def _render_model_reasoning(begruendung: object, *, stage_label: str) -> None:
    text = str(begruendung or "").strip()
    print("\n  [MODEL REASONING]")
    if text:
        _block(f"«{text}»")
    else:
        print("    (no reasoning returned)")


def _render_cited_passages(
    evidenz: List[Dict[str, Any]],
    *,
    stage_label: str,
    compact_stage: bool = False,
) -> None:
    """Human-readable evidence cards extracted from the LLM JSON."""
    print("\n  [CITED PASSAGES — evidence extracted from the reports]")
    if not evidenz:
        if compact_stage:
            print(
                "    Stage 1 uses compact output: a short reason only, no text citations.\n"
                "    Citations appear in Stage 2 (subtype), if the case is hemorrhagic."
            )
        else:
            print("    (no structured citations returned by the model)")
        return

    for i, entry in enumerate(evidenz, start=1):
        berichttyp = str(entry.get("berichttyp", "") or "").strip()
        feld = str(entry.get("feld", "") or "").strip()
        textstelle = str(entry.get("textstelle", "") or "").strip()
        interpretation = str(entry.get("interpretation", "") or "").strip()
        source = " · ".join(p for p in (berichttyp, feld) if p) or "clinical text"

        print(f"\n    {i}. {source}")
        if textstelle:
            print(f"       Quote:          «{textstelle}»")
        if interpretation:
            print(f"       Interpretation: {interpretation}")
        if not textstelle and not interpretation:
            print("       (empty evidence item)")


def _render_stage_result(stage: Dict[str, Any], *, stage_label: str, compact: bool = False) -> None:
    """Parsed fields + readable reasoning / evidence (not just raw JSON keys)."""
    parsed = stage.get("parsed") or {}
    print("\n  [PARSED & VALIDATED]")
    if stage_label == "Stage 1":
        print(f"    Decision:      {parsed.get('label')}  (klasse={parsed.get('klasse')})")
        print(f"    Confidence:    {parsed.get('sicherheit')}")
    else:
        print(f"    Subtype:         {parsed.get('haemorrhage_subtype')}")
        print(f"    Confidence:    {parsed.get('sicherheit')}")

    _render_model_reasoning(parsed.get("begruendung"), stage_label=stage_label)
    _render_cited_passages(
        _evidenz_from_stage(stage),
        stage_label=stage_label,
        compact_stage=compact,
    )


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
            label = rep.get("typus_label", "")
            body = str(rep.get("report_text", "") or "")
            print(f"  [{label}]  ({len(body):,} chars)")
            sections = _split_report_sections(body)
            if len(sections) > 1 or (sections and sections[0][0] != "Text"):
                for heading, section_body in sections:
                    if not section_body.strip():
                        continue
                    print(f"    · {heading}")
                    _block(section_body, limit=text_limit, indent="      ")
            else:
                _block(body, limit=text_limit)
            print("")
    else:
        _block(input_text or "(no report text)", limit=text_limit)
    _pause(pause)

    # ---- STEP 2: evidence block ------------------------------------------ #
    _step(2, "Evidence presented to the LLM")
    _explain(
        "These report sections are concatenated and forwarded as-is — no manual\n"
        "feature engineering and no NLP preprocessing. The model reads this raw text."
    )
    _render_llm_input_evidence(reports, input_text, limit=text_limit)
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
    _render_stage_result(stage1, stage_label="Stage 1", compact=True)
    _explain("The parser validates and normalises the model output before anything uses it.")
    _pause(pause)

    p1 = stage1.get("parsed", {})
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
        _render_stage_result(stage2, stage_label="Stage 2", compact=False)
        p2 = stage2.get("parsed", {})
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


def _first_case_id(preds: pd.DataFrame, mask: "pd.Series") -> Optional[str]:
    hits = preds[mask]
    if hits.empty:
        return None
    return str(hits.iloc[0].get("case_id", "")).strip() or None


def _excluded_pid_mask(preds: pd.DataFrame, exclude_pids: Optional[frozenset[str]]) -> "pd.Series":
    if not exclude_pids or "excel_pid" not in preds.columns:
        return pd.Series(False, index=preds.index)
    norm = {str(p).strip() for p in exclude_pids}
    return preds["excel_pid"].astype(str).str.strip().isin(norm)


def _autopick_negative_from_predictions(
    preds: pd.DataFrame,
    exclude_pids: Optional[frozenset[str]] = None,
) -> Optional[str]:
    """
    Pick a *correct* nicht_hämorrhagisch case (true negative) so Stage 2 is skipped
    AND the demo never shows a misclassification (e.g. a false negative). Requires
    the reference label to confirm correctness; falls back to any predicted
    nicht_hämorrhagisch case only if no reference labels are available at all.
    Patients in ``exclude_pids`` are never selected.
    """
    if preds.empty or "label" not in preds.columns:
        return None
    keep = ~_excluded_pid_mask(preds, exclude_pids)
    base = (
        keep
        & preds.get("status", "").astype(str).str.strip().eq("success")
        & preds["label"].astype(str).str.strip().str.lower().eq("nicht_hämorrhagisch")
    )
    if "reference_label_status" in preds.columns:
        ref = preds["reference_label_status"].astype(str).str.strip().str.lower()
        # Prefer a verified true negative; never knowingly show a false negative.
        picked = _first_case_id(preds, base & ref.eq("non_hemorrhagic"))
        if picked:
            return picked
        # If reference labels exist but none are a confirmed TN, do NOT fall back to
        # an unverified (possibly wrong) case — signal "no clean negative found".
        if ref.ne("").any():
            return None
    return _first_case_id(preds, base)


def _select_polarity_case(
    cases: List[ClinicalCase],
    preds: Optional[pd.DataFrame],
    *,
    kind: str,
    case_id: Optional[str],
    exclude_pids: Optional[frozenset[str]] = None,
) -> Optional[ClinicalCase]:
    by_id = {c.case_id: c for c in cases}
    if case_id:
        # Explicit selection wins — even if otherwise excluded.
        return by_id.get(case_id)

    excluded = {str(p).strip() for p in (exclude_pids or frozenset())}

    if preds is not None:
        picked = (
            _autopick_from_predictions(preds, exclude_pids)
            if kind == "positive"
            else _autopick_negative_from_predictions(preds, exclude_pids)
        )
        if picked and picked in by_id:
            return by_id[picked]

    # Fallback heuristic when no predictions are available.
    want_bleeding = kind == "positive"
    for c in cases:
        if str(c.excel_pid).strip() in excluded:
            continue
        if _has_bleeding_hint(c) == want_bleeding:
            return c
    for c in cases:
        if str(c.excel_pid).strip() not in excluded:
            return c
    return None


def generate_snapshot(
    *,
    kind: str,
    out_path: Path,
    case_id: Optional[str] = None,
    live: bool = False,
    predictions_path: Optional[Path] = None,
    reports_path: Optional[Path] = None,
    exclude_pids: Optional[frozenset[str]] = None,
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

    excluded = DEMO_EXCLUDED_PIDS if exclude_pids is None else exclude_pids
    if excluded and not case_id:
        print(f"Excluding patient(s) from auto-pick: {sorted(excluded)}")
    case = _select_polarity_case(
        cases, preds, kind=kind, case_id=case_id, exclude_pids=excluded
    )
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

    # Correctness check vs reference — so we never present a misclassification.
    ref_status = _reference_status_for_case(preds, case.case_id)
    if ref_status:
        expected = "hemorrhagic" if kind == "positive" else "non_hemorrhagic"
        correct = ref_status == expected
        print(f"  reference_label_status = {ref_status}   ->  {'CORRECT ✓' if correct else 'MISMATCH ✗'}")
        if not correct:
            print(
                "  [!] WARNING: this case is NOT a verified "
                f"{'true positive' if kind == 'positive' else 'true negative'}.\n"
                "      Pick a clean case explicitly with --case-id, or check the predictions CSV."
            )
    else:
        print("  reference_label_status = (no reference label available for this case)")
    return 0


def _reference_status_for_case(preds: Optional[pd.DataFrame], case_id: str) -> str:
    if preds is None or "reference_label_status" not in preds.columns:
        return ""
    row = preds[preds.get("case_id", "").astype(str).str.strip() == str(case_id).strip()]
    if row.empty:
        return ""
    return str(row.iloc[0].get("reference_label_status", "")).strip().lower()


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
    parser.add_argument(
        "--exclude-pid",
        action="append",
        default=None,
        metavar="PID",
        help="excel_pid to skip during auto-pick (repeatable). Adds to the built-in exclusions.",
    )
    parser.add_argument("--live", action="store_true", help="Call the real LLM when generating a snapshot")
    parser.add_argument("--predictions", type=Path, default=None, help="Predictions CSV (replay source)")
    parser.add_argument("--reports", type=Path, default=None, help="Reports Excel input")
    args = parser.parse_args(argv)

    exclude_pids = DEMO_EXCLUDED_PIDS | frozenset(args.exclude_pid or ())

    if args.snapshot_positive:
        return generate_snapshot(
            kind="positive",
            out_path=POSITIVE_SNAPSHOT,
            case_id=args.case_id,
            live=args.live,
            predictions_path=args.predictions,
            reports_path=args.reports,
            exclude_pids=exclude_pids,
        )
    if args.snapshot_negative:
        return generate_snapshot(
            kind="negative",
            out_path=NEGATIVE_SNAPSHOT,
            case_id=args.case_id,
            live=args.live,
            predictions_path=args.predictions,
            reports_path=args.reports,
            exclude_pids=exclude_pids,
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
