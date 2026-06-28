# LLM Extraction Demo — Guide

A polished, interactive walkthrough that shows **how an LLM + prompt-engineering
pipeline turns an unstructured German clinical report into validated, structured
information.**

The hemorrhage task is only the worked example. The five stages below are a
**reusable blueprint** for *any* medical information-extraction task (e.g. a
Cardiology department building a similar pipeline):

```
Clinical reports  →  Prompt Engineering  →  LLM  →  JSON Validation  →  Structured Output  →  Spreadsheet
```

Key properties for a presentation:

- **Runs instantly.** It never calls the LLM during the demo.
- **Uses real, previously captured model responses** (frozen in JSON snapshots).
- **Two examples**: one hemorrhagic (both stages run) and one non-hemorrhagic
  (Stage 2 is skipped) — so the *conditional, hierarchical* design is obvious.
- **Self-paced**: it pauses after every section (`Press ENTER to continue…`).

---

## 1. One-time setup: generate the snapshots

Snapshots are self-contained JSON files holding everything the demo needs (report
text, the exact prompts, the real raw LLM responses, the parsed results, and the
final classification). Generate them **once**, where the data and the predictions
CSV exist (typically the server after a full run):

```bash
cd ~/hemorrhage_project
source Ba_venv/bin/activate
export PROJECT_TASK=hemorrhage

# Auto-picks a clear hemorrhagic case and a clear non-hemorrhagic case,
# reusing the responses already stored in data/outputs/hemorrhage_case_predictions.csv
python3 -m src.tasks.hemorrhage.demo --snapshot-positive
python3 -m src.tasks.hemorrhage.demo --snapshot-negative
```

This writes:

```
data/demo/positive_case.json
data/demo/negative_case.json
```

> **Correctness guarantee:** the auto-picker only selects cases that the model got
> **right** (verified against the reference label): a true positive for the positive
> demo and a true negative for the negative demo. It will never auto-pick a
> misclassification (e.g. a false negative). The command prints the chosen case's
> `reference_label_status` and a `CORRECT ✓` / `MISMATCH ✗` line so you can confirm.
> If no verified clean case exists, use `--case-id` to choose one explicitly.

These two files are portable — copy them to any machine (e.g. your laptop) and the
demo will run there with **no data, no predictions CSV and no LLM**.

### Choosing a specific case

```bash
python3 -m src.tasks.hemorrhage.demo --snapshot-positive --case-id case_..._..._...
python3 -m src.tasks.hemorrhage.demo --snapshot-negative --case-id case_..._..._...
```

### Excluding a patient from auto-pick

Some reference labels are clinically unreliable (the model may "agree" with a wrong
label, so a case looks like a correct true negative while actually being a missed
bleeding). Such patients are skipped via a built-in exclusion list
(`DEMO_EXCLUDED_PIDS` in `demo.py`). To exclude more on the fly:

```bash
python3 -m src.tasks.hemorrhage.demo --snapshot-negative --exclude-pid 10206120
#   --exclude-pid is repeatable and adds to the built-in exclusions
```

### Capturing a fresh response live (slow; needs the LLM server)

If a case has no stored response yet, add `--live` to call the model once while
generating the snapshot (the demo itself still never calls the LLM):

```bash
export LLM_PROVIDER=usz_api LLM_TEMPERATURE=0
python3 -m src.tasks.hemorrhage.demo --snapshot-positive --live
```

---

## 2. Run the demo

### Interactive menu (recommended for presenting)

```bash
python3 -m src.tasks.hemorrhage.demo
```

```
============================================================
  HEMORRHAGE DEMO  ·  LLM extraction pipeline
============================================================
  Choose demonstration:

    [1]  Positive hemorrhagic case
    [2]  Negative non-hemorrhagic case
    [3]  Run both
    [q]  Quit

  >
```

### Direct (skip the menu)

```bash
python3 -m src.tasks.hemorrhage.demo --positive
python3 -m src.tasks.hemorrhage.demo --negative
python3 -m src.tasks.hemorrhage.demo --both
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--no-pause` | Don't wait for ENTER between sections (run straight through) |
| `--full` | Show the full prompts / report text instead of readable excerpts |

---

## 3. What each step shows

For every case the demo presents seven steps, each ending with a pause:

| Step | Title | Purpose |
|------|-------|---------|
| 1 | Original clinical reports | The raw, unstructured German documents the pipeline starts from. |
| 2 | Evidence presented to the LLM | The exact text block forwarded to the model — no NLP preprocessing. |
| 3 | Stage 1 prompt | The engineered SYSTEM + USER prompt for the **binary** decision (excerpt). |
| 4 | Real LLM response → validated JSON | The raw model output, then the parsed/validated fields. |
| 5 | Stage 2 prompt | The narrower **subtype** prompt — *only runs if Stage 1 found a hemorrhage*. |
| 6 | Real subtype response → validated JSON | Raw subtype output, then parsed fields. |
| 7 | Final structured output | Final classification + the spreadsheet one-hot column + the pipeline summary. |

**Negative case:** after Step 4 the demo prints `STAGE 2 SKIPPED — No hemorrhage
detected` and jumps straight to the final output. This visibly demonstrates the
efficiency of the hierarchical design (the second, more expensive LLM call is
avoided).

---

## 4. Expected output (shape)

```
============================================================
  CASE  case_…__…__…
  POSITIVE · hämorrhagisch
============================================================

============================================================
STEP 1  ·  Original clinical reports
============================================================

The pipeline receives completely unstructured German clinical documentation.

  [01 Operationsbericht]
    …
  [02 Eintrittsbericht]
    …

Press ENTER to continue…
```

…and at the end:

```
  ————————————————————————————————————
  Final Classification

  Hemorrhagic:         YES
  Subtype:             nicht_akut
  Confidence:          hoch
  Spreadsheet column:  hämorrhagisch nicht akut
  ————————————————————————————————————
```

---

## 5. Adapting this to another extraction task (blueprint)

To reuse the pattern for a different domain:

1. **Define the target labels** (your equivalent of klasse / subtype).
2. **Write the prompts** (`prompts/*.txt`): a SYSTEM prompt with the rules + a strict
   JSON schema, and a USER prompt that embeds the raw evidence text.
3. **Write a parser** that validates and normalises the model's JSON before use.
4. (Optional) **Split into stages** — a cheap first decision, then a narrower second
   prompt only when needed — to save tokens/latency, exactly like Stage 1 → Stage 2 here.
5. **Capture responses once** and present from snapshots.

The demo code (`src/tasks/hemorrhage/demo.py`) is intentionally separated from the
production pipeline and can be copied as a starting point.

---

## 6. Files involved

| File | Role |
|------|------|
| `src/tasks/hemorrhage/demo.py` | Interactive demo + snapshot generation (this guide's commands) |
| `src/tasks/hemorrhage/demo_extraction.py` | Lower-level trace builder reused by the demo |
| `data/demo/positive_case.json` | Frozen hemorrhagic example (generated) |
| `data/demo/negative_case.json` | Frozen non-hemorrhagic example (generated) |
| `docs/demo/DEMO_GUIDE.md` | This guide |

The demo **does not** modify the production pipeline, prompts, or parser — it only
reads them, so what you see is exactly what runs in production.
