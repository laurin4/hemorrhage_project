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
python3 -m src.tasks.hemorrhage.run_case_pipeline
```

**Outputs:**

- `data/inspection/` — structural + label analytics
- `data/outputs/hemorrhage_case_predictions.csv` — one row per case
- `data/outputs/hemorrhage_prediction_review.csv` — unified qualitative review table
- `data/outputs/hemorrhage_confusion_review.csv` — compact TP/TN/FP/FN overview

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

```bash
wc -l data/outputs/hemorrhage_false_negative_review.csv
wc -l data/outputs/hemorrhage_false_positive_review.csv
head -5 data/outputs/hemorrhage_false_negative_review.csv
```

Summary: `data/outputs/hemorrhage_prediction_review_summary.txt`

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
| `--all-cases` | Process ALL cases (disable default cohort filter) |
| `--include-verify-only` | Add `verify_only` cases to the labeled cohort |

### Inference cohort (default = labeled binary only)

By default the pipeline runs **only on binary-labeled cases** — reference status `hemorrhagic` or `non_hemorrhagic` — for valid evaluation and faster runtime.

- Excluded by default: `verify_only`, `unknown`, `inconsistent` (counts reported in startup + summary as `excluded_by_status`).
- `--include-verify-only` additionally processes `verify_only` cases.
- `--all-cases` disables filtering entirely.
- If **no reference is available**, filtering is skipped (cannot determine status) and all cases are processed with a warning (`cohort_mode="all_cases (no reference available)"`).
- Reference status is derived from the spreadsheet label cells via `io.reference_lookup.reference_binary_status` (single source of truth, shared with the review/eval `derive_reference_status`).

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
