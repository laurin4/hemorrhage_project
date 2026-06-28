# Hemorrhage task — case-centric pipeline

## How to run on the server (full workflow)

```bash
cd ~/hemorrhage_project
source Ba_venv/bin/activate
export PROJECT_TASK=hemorrhage

# Verify raw files
ls -la data/raw/NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx
ls -la data/raw/260507_CCM_DAVF.xlsx

python3 -m src.tasks.hemorrhage.inspect_data
python3 -m src.tasks.hemorrhage.analyze_reference_labels
python3 -m src.tasks.hemorrhage.run_case_pipeline --dry-run --limit 5

export LLM_PROVIDER=usz_api
export LLM_TEMPERATURE=0
python3 -m src.tasks.hemorrhage.run_case_pipeline --limit 5
python3 -m src.tasks.hemorrhage.run_case_pipeline                 # ALL patients (default)
python3 -m src.tasks.hemorrhage.run_case_pipeline --labeled-only  # eval cohort only

# Merge the case classifications into the patient/case spreadsheet
python3 -m src.tasks.hemorrhage.merge_classifications
```

**Outputs:**

- `data/inspection/` — structural + label analytics
- `data/outputs/hemorrhage_case_predictions.csv` — one row per case
- `data/outputs/hemorrhage_prediction_review.csv` — unified qualitative review table
- `data/outputs/hemorrhage_confusion_review.csv` — compact TP/TN/FP/FN overview
- `data/outputs/NCH_cavernom_eingeblutet_classified.xlsx` — the template spreadsheet with one-hot class columns filled

## Prediction review export (qualitative analysis)

```bash
python3 -m src.tasks.hemorrhage.build_prediction_review
python3 -m src.tasks.hemorrhage.build_prediction_review --only-mismatches
python3 -m src.tasks.hemorrhage.build_prediction_review --only-labeled --limit 20
python3 -m src.tasks.hemorrhage.build_prediction_review --only-fn
python3 -m src.tasks.hemorrhage.build_prediction_review --only-fp
```

Combines predictions, reference labels, reasoning, evidence, and compact case previews.  
Preliminary comparison only — **not final evaluation**.

**Automatic detailed error exports** (for manual validation):

- `data/outputs/hemorrhage_false_negative_review.csv` — all FN cases, full detail
- `data/outputs/hemorrhage_false_positive_review.csv` — all FP cases, full detail

**Final-target exports** (split of all hemorrhagic predictions by clinical relevance, for manual clinical review):

- `data/outputs/hemorrhage_clinically_relevant_cases.csv` — `label==hämorrhagisch` AND `predicted_haemorrhage_subtype != historisch` (akut/nicht_akut)
- `data/outputs/hemorrhage_historical_cases.csv` — `label==hämorrhagisch` AND `predicted_haemorrhage_subtype == historisch`
- `data/outputs/hemorrhage_final_target_summary.csv` — counts (`metric,count`): `total_processed_cases`, `clinically_relevant_hemorrhage`, `historical_hemorrhage`, `non_hemorrhagic`, `prediction_missing`, `parse_failed`, `llm_failed`

Both split exports carry the **full review columns plus `final_target_label`**. Invariant: `clinically_relevant + historical = all hemorrhagic predictions` (every hemorrhagic prediction appears in exactly one export). The split is computed over **all** predictions, independent of `--only-labeled` / `--only-mismatches` filters.

```bash
wc -l data/outputs/hemorrhage_false_negative_review.csv
wc -l data/outputs/hemorrhage_false_positive_review.csv
cat data/outputs/hemorrhage_final_target_summary.csv
head -5 data/outputs/hemorrhage_historical_cases.csv
```

Summary: `data/outputs/hemorrhage_prediction_review_summary.txt`

## Demo — watch the LLM extraction work (proof of concept)

A polished, interactive walkthrough that shows how the prompt-based pipeline turns
unstructured German clinical text into validated, structured output (free-text →
prompt engineering → LLM → JSON validation → structured output → spreadsheet). It
**runs instantly and never calls the LLM** during a presentation: it replays real,
previously captured responses frozen in `data/demo/`. Full guide:
[`docs/demo/DEMO_GUIDE.md`](../../../docs/demo/DEMO_GUIDE.md).

```bash
# Interactive menu: [1] positive  [2] negative  [3] both  [q] quit
python3 -m src.tasks.hemorrhage.demo

# Direct (skip the menu)
python3 -m src.tasks.hemorrhage.demo --positive
python3 -m src.tasks.hemorrhage.demo --negative
python3 -m src.tasks.hemorrhage.demo --both
#   --no-pause  run straight through (no ENTER waits) · --full  show full prompts
```

It demonstrates two cases side by side — a hemorrhagic one (both stages run) and a
non-hemorrhagic one (`STAGE 2 SKIPPED`) — so the conditional, hierarchical design is
obvious.

### Generate the snapshots once (where data + predictions exist, e.g. the server)

```bash
python3 -m src.tasks.hemorrhage.demo --snapshot-positive   # → data/demo/positive_case.json
python3 -m src.tasks.hemorrhage.demo --snapshot-negative   # → data/demo/negative_case.json
#   pick a specific case with --case-id <case_id>; add --live to capture a fresh response
```

Each snapshot JSON is self-contained (report text, both prompts, both raw LLM
responses, parsed results, final class), so you can copy it to a laptop and present
in a meeting room with zero dependencies — no server, no Excel files.

> The lower-level `python3 -m src.tasks.hemorrhage.demo_extraction` (single-case
> narration with `--snapshot` / `--from-snapshot` / `--replay`) still exists and is
> reused under the hood; prefer `... .demo` for presentations.

---

## Preliminary evaluation (quantitative metrics + plots)

After full inference and review export:

```bash
python3 -m src.tasks.hemorrhage.evaluate_predictions
python3 -m src.tasks.hemorrhage.evaluate_predictions --include-verify-as-negative
```

**Outputs** (`data/evaluation/`):

- `hemorrhage_metrics_summary.csv` — machine-readable metrics (raw floats)
- `hemorrhage_metrics_summary.txt` / `.md` — human-readable reports for meetings / thesis notes
- `hemorrhage_confusion_matrix.csv` — aggregated confusion matrix
- `hemorrhage_error_cases.csv` — FP/FN and labeled pipeline failures
- `hemorrhage_subtype_distribution.csv` — predicted subtype counts among hämorrhagisch predictions (descriptive)
- `hemorrhage_subtype_by_reference_status.csv` — predicted subtype × reference_status crosstab (descriptive)
- `plots/` — confusion matrix, distributions, confidence by correctness, `predicted_haemorrhage_subtype_distribution.png`, `subtype_by_reference_status.png`

**Methodology:** Preliminary evaluation on labeled subset only. Verify_Vaskulär-only cases are **excluded** from default performance metrics (conservative). Use `--include-verify-as-negative` for exploratory sensitivity analysis. **Subtype analysis is descriptive only** (no validated reference subtype labels yet); binary metrics are unaffected by subtype.

Inspect:

```bash
cat data/evaluation/hemorrhage_metrics_summary.txt
cat data/evaluation/hemorrhage_confusion_matrix.csv
ls -lh data/evaluation/plots/
```

---

## Phase 1 — Case-level inference (prototype)

```bash
python3 -m src.tasks.hemorrhage.run_case_pipeline [OPTIONS]
```

| Option | Role |
|--------|------|
| `--dry-run` | Build prompts only; `status=dry_run` |
| `--limit N` | First N cases (applied **after** cohort filter) |
| `--case-id ID` | Single case (bypasses cohort filter) |
| `--output PATH` | Custom CSV path |
| `--reports` / `--reference` | Override Excel paths |
| `--labeled-only` | Restrict to the binary-labeled evaluation cohort |
| `--include-verify-only` | With `--labeled-only`, also add `verify_only` cases |

### Inference cohort (default = ALL patients)

By default the pipeline classifies **every patient**, including `verify_only` (Verify_Vaskulär), `unknown`, and unlabeled cases. They are run through the same two-stage structure; cases without a binary reference simply have no TP/TN/FP/FN.

- `--labeled-only` restricts to the binary-labeled evaluation cohort (reference status `hemorrhagic` or `non_hemorrhagic`) for valid evaluation / faster runtime. Excluded then: `verify_only`, `unknown`, `inconsistent` (reported as `excluded_by_status`).
- `--include-verify-only` (with `--labeled-only`) additionally processes `verify_only` cases.
- Reference status is derived from the spreadsheet label cells via `io.reference_lookup.reference_binary_status` (single source of truth, shared with the review/eval `derive_reference_status`).

### Classification merge (patient/case spreadsheet)

`python3 -m src.tasks.hemorrhage.merge_classifications` fills a patient/case template
(`data/raw/NCH_cavernom_eingeblutet.xlsx`, override via `HEMORRHAGE_CLASSIFICATION_TEMPLATE_XLSX`) with **one-hot** final-class columns and writes a merged copy to `data/outputs/NCH_cavernom_eingeblutet_classified.xlsx` (the raw template is never modified).

- The template has **one row per report**; each case `(excel_pid, excel_opdat, opber_fallnr)` is classified once and broadcast onto all its report rows.
- One-hot columns (`1` / `0`): `hämorrhagisch akut`, `hämorrhagisch nicht akut`, `hämorrhagisch historisch`, `nicht hämorrhagisch`.
- Failed (`parse_failed` / `llm_failed`), unknown-subtype, or unmatched cases leave all four columns **blank** and record a reason in `klassifikation_status` (so an empty cell is never confused with a real `0`).
- Key matching uses the same normalization as the reports loader (`excel_pid`/`opber_fallnr` stringified, `excel_opdat` → ISO date). Unmatched template rows are listed in `data/outputs/hemorrhage_classification_unmatched_rows.csv`; counts go to `hemorrhage_classification_merge_summary.txt`.

### Two-stage hierarchical inference (architecture)

Inference is split into two sequential LLM calls per case to cut token generation and avoid `ReadTimeout`s:

- **Stage 1 — binary** (`prompts/hemorrhage_binary_classification.txt`): decides only `klasse` 0/1 (`nicht_hämorrhagisch` vs `hämorrhagisch`). No subtype. **Compact output** — `{klasse, label, sicherheit, kurzbegruendung}`, no evidence list (the parser accepts `kurzbegruendung` as `begruendung`).
- **Stage 2 — subtype** (`prompts/hemorrhage_subtype_classification.txt`): runs **only when `klasse=1`**; decides only `haemorrhage_subtype` ∈ {`historisch`, `nicht_akut`, `akut`} with `{sicherheit, begruendung, evidenz}` (evidence ≤3 items). It assumes the hemorrhage already exists and must not reconsider it.

**Final binary definition (Stage 1):** `klasse=1` requires explicit/clinically relevant hemorrhage evidence (Blutung/Einblutung/geblutet/hämorrhagisch/Hämatom/Hämatomevakuation/clear bleeding context). The following alone are **not** sufficient for `klasse=1`: Kavernom/CCM, DAVF/AVM/vascular lesion, epilepsy, resection/operation, vascular verification, or a lesion diagnosis without bleeding wording. Historical hemorrhage is still `klasse=1` (subtype decided in Stage 2). **Final subtype definition (Stage 2):** `historisch` = only a previous/historical event; `nicht_akut` = relevant in the current case but not acute; `akut` = acute/fresh/subacute bleeding or acute hemorrhage-related treatment (incl. hematoma evacuation).
- The runner (`process_single_case`) merges Stage 1 + Stage 2 into one prediction row; the CSV schema is unchanged.
- Non-hemorrhagic cases terminate after Stage 1, so only positives pay for subtype reasoning (`subtype_stage_status=skipped`).
- The original combined prompt (`prompts/hemorrhage_case_classification.txt`) and `parse_hemorrhage_response` remain available for single-call use/tests, but the pipeline uses the two-stage path.

New logging/diagnostic columns: `binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length`. `raw_response_length` is the sum of both stages' raw output; `prompt_length_chars` is the sum of both stage prompts.

### Two-level classification (supervisor clarification)

- **Level 1:** `klasse=0 → nicht_hämorrhagisch`, `klasse=1 → hämorrhagisch`.
- **Level 2 (only if `klasse=1`):** `haemorrhage_subtype` ∈ {`akut`, `nicht_akut`, `historisch`} (mandatory when hemorrhagic).
  - `nicht_hämorrhagisch` → `haemorrhage_subtype = null`.
  - hämorrhagisch but subtype missing/unclear → parser fallback `haemorrhage_subtype = "unbekannt"` + uncertainty reason (no parse failure). The model itself must not emit `unbekannt`.
- **Historical hemorrhage is still hemorrhage.** A past/remote bleed → `klasse=1`, `label="hämorrhagisch"`, `subtype="historisch"` (NEVER `klasse=0`).
  - `akut` = current acute/fresh bleeding event.
  - `nicht_akut` = current-case hemorrhagic finding that is not acute (e.g. chronic lesion).
  - `historisch` = previous/past/old hemorrhage in history (incl. "Status nach Blutung"); even if it was acute at the time.
- **`Verify_Vaskulär` is metadata only**, never a class label. It must not influence the model decision and is excluded from binary TP/TN/FP/FN (unless `--include-verify-as-negative` sensitivity mode).
- **Binary evaluation is unchanged** (hemorrhagic vs non_hemorrhagic; historical counts as positive). **Subtype analysis is descriptive only** — no validated reference subtype labels exist yet, so subtype accuracy is not computed.

JSON schema returned by the LLM:

```json
{
  "klasse": 0,
  "label": "nicht_hämorrhagisch",
  "haemorrhage_subtype": null,
  "sicherheit": "niedrig",
  "begruendung": "...",
  "evidenz": [{"berichttyp": "...", "feld": "...", "textstelle": "...", "interpretation": "..."}],
  "historische_blutung_erwaehnt": false,
  "historische_blutung_als_aktuell_gewertet": false,
  "unsicherheitsgruende": []
}
```

**No keyword prefilter.** Incomplete cases are sent to LLM.

### LLM robustness / stability

The pipeline never crashes on a single slow/failed LLM call:

- `HEMORRHAGE_LLM_TIMEOUT_SECONDS` — per-call read timeout (default **240**).
- `HEMORRHAGE_LLM_MAX_RETRIES` — automatic retries (default **1**), only on `ReadTimeout` / `Timeout` / `ConnectionError`, with a 5 s wait between attempts.
- On exhausted retries the case is recorded as `status=llm_failed` with `error_message="<ExcType> after <timeout> seconds (retries=N)"` and the run continues.
- Predictions are written **incrementally** (header + flush per case), so completed rows survive an interrupted run.
- Debug columns `prompt_length_chars`, `structured_case_text_length` and `raw_response_length` help identify oversized cases / verbose responses that cause timeouts.
- The prompts enforce compact output (≤3 evidenz items, `textstelle` ≤200 chars, `interpretation` 1 sentence, `begruendung` ≤2 sentences) to cut generation time without changing classification quality.
- **Two-stage inference** is the primary timeout mitigation: non-hemorrhagic cases finish after the light binary call and never trigger subtype reasoning. If Stage 2 alone times out/fails, the case stays `hämorrhagisch` with `haemorrhage_subtype="unbekannt"` (`subtype_stage_status=llm_failed`) — Stage 1 is preserved, not lost.
- A per-case log line is emitted before each call: `[i/total] <case_id> text_length=… binary_prompt_length=… reports=…`.
- End-of-run summary prints `successful_cases` / `parse_failed_cases` / `llm_failed_cases`.

Delirium pipeline: `src.pipeline.run_pipeline` — **not used** for hemorrhage.

**Internal modules:** `src/tasks/hemorrhage/io/load_cases.py`, `reference_lookup.py`  
**No new Python packages** required — `wheelhouse_linux` is only for external deps, not internal imports.

---

## Phase 0b — Inspection

```bash
python3 -m src.tasks.hemorrhage.inspect_data
python3 -m src.tasks.hemorrhage.analyze_reference_labels
```

Reports: `NCH_pidlist_...xlsx` | Reference: `260507_CCM_DAVF.xlsx`

---

## Phase 0 — Case build from CSV

```bash
python3 -m src.tasks.hemorrhage.build_cases --input data/raw/reports.csv
```
