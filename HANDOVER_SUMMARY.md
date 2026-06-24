# Handover Summary

> **Active task:** Hemorrhage — **case-centric** architecture (Phase 0).  
> **Full engineering doc:** `TECHNICAL_HANDOVER.md`  
> **Delirium legacy pipeline:** still in repo but isolated — see `src/tasks/delirium/BOUNDARIES.md`.

---

## Hemorrhage project (current)

### Purpose (target)

Classify clinical **cases** from OP / Eintritts / Austritts reports using structured evidence + local LLM (later phases) with a **two-level target** (supervisor clarification):

- **Level 1:** `klasse=0 → nicht_hämorrhagisch`, `klasse=1 → hämorrhagisch`.
- **Level 2 (only if `klasse=1`):** `haemorrhage_subtype` ∈ {`akut`, `nicht_akut`, `historisch`}; `null` for non-hemorrhagic; parser fallback `"unbekannt"` if hemorrhagic but subtype unclear.
- **Historical hemorrhage is still hemorrhage:** past/remote bleed → `klasse=1` + `subtype="historisch"` (NEVER `klasse=0`).
- **`Verify_Vaskulär` is metadata/reference only**, NOT a ground-truth class, and is excluded from binary TP/TN/FP/FN by default.
- **Binary evaluation is unchanged** (historical counts as positive); **subtype analysis is descriptive only** (no validated reference subtype labels yet).

### Final clinical working definition

- **klasse=1 / hämorrhagisch** requires explicit/clinically relevant hemorrhage evidence: Blutung, Einblutung/eingeblutet, geblutet, hämorrhagisch/hemorrhagic, Hämatom/Hematom, Hämatomevakuation, or a clear hemorrhage-related treatment context.
- **klasse=0 / nicht_hämorrhagisch** = no such evidence. The following alone are **NOT** sufficient for klasse=1: Kavernom/CCM, DAVF/AVM/vascular lesion, epilepsy/seizures, resection/operation, vascular verification, lesion diagnosis without bleeding wording.
- **Subtype** (only if klasse=1): `historisch` (only a previous/historical event), `nicht_akut` (relevant now but not acute), `akut` (acute/fresh/subacute or acute hemorrhage-related treatment incl. hematoma evacuation).
- Historical hemorrhage stays klasse=1 + subtype=historisch. `Verify_Vaskulär` is metadata only.

### Inference cohort (default = ALL patients)

Default runs now classify **every patient** (incl. `verify_only` / Verify_Vaskulär, `unknown`, unlabeled) through the same two-stage structure; unlabeled cases just have no TP/TN/FP/FN. CLI flag `--labeled-only` restricts to the binary-labeled evaluation cohort (`hemorrhagic`/`non_hemorrhagic`) for valid evaluation + faster runtime; with it, `--include-verify-only` also adds `verify_only`. `--case-id` bypasses the filter. Status is derived by `io.reference_lookup.reference_binary_status`. (`--all-cases` still accepted as a hidden alias of the default.)

### Classification merge → patient/case spreadsheet

`python3 -m src.tasks.hemorrhage.merge_classifications` fills a template (`data/raw/NCH_cavernom_eingeblutet.xlsx`; env `HEMORRHAGE_CLASSIFICATION_TEMPLATE_XLSX`) with one-hot final-class columns and writes `data/outputs/NCH_cavernom_eingeblutet_classified.xlsx` (raw never mutated). Template is **one row per report**; each case `(excel_pid, excel_opdat, opber_fallnr)` is classified once and broadcast onto all its report rows. Columns (`1`/`0`): `hämorrhagisch akut`, `hämorrhagisch nicht akut`, `hämorrhagisch historisch`, `nicht hämorrhagisch`. Failed / unknown-subtype / unmatched rows stay blank with a reason in `klassifikation_status`. Logic + tests: `src/tasks/hemorrhage/export/classification_merge.py`, `tests/test_classification_merge.py`.

### Inference architecture — two-stage (current)

To reduce token generation and `ReadTimeout`s, each case runs **two sequential LLM calls** instead of one combined call:

- **Stage 1 — binary** (`prompts/hemorrhage_binary_classification.txt`): only `klasse` 0/1, compact JSON `{klasse, label, sicherheit, kurzbegruendung}` (no evidence list).
- **Stage 2 — subtype** (`prompts/hemorrhage_subtype_classification.txt`): only when `klasse=1`; only `haemorrhage_subtype` (historisch/nicht_akut/akut) with `{sicherheit, begruendung, evidenz}` (≤3 items), assuming hemorrhage exists.
- The runner merges both stages into one row; CSV schema unchanged. Non-hemorrhagic cases skip Stage 2 (`subtype_stage_status=skipped`).
- New columns: `binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length`.
- The old combined prompt (`prompts/hemorrhage_case_classification.txt`) + `parse_hemorrhage_response` are kept for single-call/testing but are no longer the pipeline path.

### Case definition

```text
Case = (excel_pid, excel_opdat, opber_fallnr)
```

Reports per case (0–3 allowed):

- `01` Operationsbericht
- `02` Eintrittsbericht
- `03` Austrittsbericht

**Incomplete cases are expected** — the pipeline must not assume all three exist.

### Phase 1 — Case-level LLM inference (prototype)

```bash
python3 -m src.tasks.hemorrhage.run_case_pipeline --dry-run --limit 5
python3 -m src.tasks.hemorrhage.run_case_pipeline --limit 5
python3 -m src.tasks.hemorrhage.run_case_pipeline
```

Output: `data/outputs/hemorrhage_case_predictions.csv` (one row per case).

No keyword prefilter. Delirium `run_pipeline.py` **unchanged**.

Case loading lives under `src/tasks/hemorrhage/io/` (`load_cases.py`, `reference_lookup.py`).  
**No additional Python packages** are required for this pipeline — only internal imports.  
`wheelhouse_linux` is relevant for external dependency installation on the server, not for internal module paths.

### How to run on the server (copy-paste, full workflow)

This is the complete end-to-end sequence. Run every command from the project root.

```bash
cd ~/hemorrhage_project
source Ba_venv/bin/activate
export PROJECT_TASK=hemorrhage

# --- LLM connection + behaviour ---
export LLM_PROVIDER=usz_api            # or "ollama"
export USZ_LLM_URL=http://localhost:8100/generate
export LLM_TEMPERATURE=0               # reproducible
export LLM_TOP_P=1

# --- LLM robustness / stability (recommended) ---
export HEMORRHAGE_LLM_TIMEOUT_SECONDS=240   # per-call read timeout (default 240)
export HEMORRHAGE_LLM_MAX_RETRIES=1         # auto-retry on timeout/connection error (default 1)

# 1. Verify raw files exist
ls -la data/raw/NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx
ls -la data/raw/260507_CCM_DAVF.xlsx

# 2. Structural inspection
python3 -m src.tasks.hemorrhage.inspect_data

# 3. Reference label analytics
python3 -m src.tasks.hemorrhage.analyze_reference_labels

# 4. Dry-run prompts (no LLM) — sanity-check prompt assembly
python3 -m src.tasks.hemorrhage.run_case_pipeline --dry-run --limit 5

# 5. Limited LLM pilot (recommended before the full run)
#    Default = ALL patients (verify_only/unknown/unlabeled included)
python3 -m src.tasks.hemorrhage.run_case_pipeline --limit 10

# 6. Full case inference (one row per case, written incrementally)
python3 -m src.tasks.hemorrhage.run_case_pipeline
#    Cohort variants:
python3 -m src.tasks.hemorrhage.run_case_pipeline --labeled-only                      # eval cohort
python3 -m src.tasks.hemorrhage.run_case_pipeline --labeled-only --include-verify-only

# 7. Build review exports (also writes FN/FP detailed reviews + confusion review)
python3 -m src.tasks.hemorrhage.build_prediction_review

# 8. Preliminary evaluation (metrics + readable reports + subtype tables + plots)
python3 -m src.tasks.hemorrhage.evaluate_predictions
#    optional exploratory sensitivity (verify_only treated as non_hemorrhagic):
python3 -m src.tasks.hemorrhage.evaluate_predictions --include-verify-as-negative

# 9. Merge classifications into the patient/case spreadsheet (one-hot columns)
python3 -m src.tasks.hemorrhage.merge_classifications

# 10. Inspect key results
cat  data/evaluation/hemorrhage_metrics_summary.txt
cat  data/evaluation/hemorrhage_subtype_distribution.csv
head -20 data/outputs/hemorrhage_confusion_review.csv
cat  data/outputs/hemorrhage_classification_merge_summary.txt
ls -lh data/evaluation/plots/
```

`--limit N` (first N cases, applied after cohort filter), `--case-id ID` (single case, bypasses
cohort filter), `--dry-run` (no LLM), `--output PATH`, `--labeled-only` (binary eval cohort), and
`--include-verify-only` (with `--labeled-only`, add verify_only cases) are available on `run_case_pipeline`.

Expected outputs:

- `data/inspection/` — schema, merge, label analytics
- `data/outputs/hemorrhage_case_predictions.csv` — case predictions (incl. `klasse`, `label`, `haemorrhage_subtype`, two-stage columns `binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length`, and debug columns `prompt_length_chars`, `structured_case_text_length`, `raw_response_length`)
- `data/outputs/hemorrhage_parse_failures.csv` — debug rows for any `parse_failed` case
- `data/outputs/hemorrhage_prediction_review.csv` — unified qualitative review table (incl. `predicted_haemorrhage_subtype`)
- `data/outputs/hemorrhage_confusion_review.csv` — compact TP/TN/FP/FN review
- `data/outputs/hemorrhage_false_negative_review.csv` / `hemorrhage_false_positive_review.csv` — detailed FN/FP reviews
- `data/outputs/hemorrhage_clinically_relevant_cases.csv` — hemorrhagic predictions with subtype ≠ historisch (akut/nicht_akut), full review columns + `final_target_label`
- `data/outputs/hemorrhage_historical_cases.csv` — hemorrhagic predictions with subtype = historisch, full review columns + `final_target_label`
- `data/outputs/hemorrhage_final_target_summary.csv` — counts: total_processed_cases, clinically_relevant_hemorrhage, historical_hemorrhage, non_hemorrhagic, prediction_missing, parse_failed, llm_failed (clinically_relevant + historical = all hemorrhagic predictions)
- `data/outputs/hemorrhage_prediction_review_summary.txt` — review summary
- `data/evaluation/hemorrhage_metrics_summary.csv` / `.txt` / `.md` — metrics (raw + readable reports)
- `data/evaluation/hemorrhage_confusion_matrix.csv`, `hemorrhage_error_cases.csv`
- `data/evaluation/hemorrhage_subtype_distribution.csv`, `hemorrhage_subtype_by_reference_status.csv` — descriptive subtype tables
- `data/evaluation/plots/` — confusion matrix, distributions, confidence-by-correctness, `predicted_haemorrhage_subtype_distribution.png`, `subtype_by_reference_status.png`

### LLM robustness / stability (since stability update)

The pipeline never crashes on a single slow or failed LLM call:

- `HEMORRHAGE_LLM_TIMEOUT_SECONDS` (default **240**) — per-call read timeout.
- `HEMORRHAGE_LLM_MAX_RETRIES` (default **1**) — auto-retry only on `ReadTimeout` / `Timeout` / `ConnectionError`, 5 s wait between attempts.
- On exhausted retries: the case is recorded as `status=llm_failed` with a clear `error_message`, and the run **continues** with the next case.
- Predictions are written **incrementally** (header + flush per case), so completed rows survive an interrupted run.
- Per-case log line before each call: `[i/total] <case_id> text_length=… prompt_length=… reports=…`.
- End-of-run summary prints `successful_cases` / `parse_failed_cases` / `llm_failed_cases`.
- Debug columns `prompt_length_chars`, `structured_case_text_length` + `raw_response_length` help spot oversized cases / verbose responses that time out.
- The prompt enforces **compact output** (max 3 evidenz items, `textstelle` ≤200 chars, `interpretation` 1 sentence, `begruendung` ≤2 sentences, target <1500 chars) to reduce generation time without changing classification.

### Prediction review export (main qualitative artifact)

```bash
python3 -m src.tasks.hemorrhage.build_prediction_review
python3 -m src.tasks.hemorrhage.build_prediction_review --only-mismatches
python3 -m src.tasks.hemorrhage.build_prediction_review --only-labeled --limit 20
```

Also writes detailed FN/FP review CSVs automatically:

- `data/outputs/hemorrhage_false_negative_review.csv`
- `data/outputs/hemorrhage_false_positive_review.csv`

Preliminary comparison only — **not final evaluation**. Summary: `data/outputs/hemorrhage_prediction_review_summary.txt`

### Preliminary evaluation (metrics + plots)

```bash
python3 -m src.tasks.hemorrhage.evaluate_predictions
python3 -m src.tasks.hemorrhage.evaluate_predictions --include-verify-as-negative
```

Outputs: `data/evaluation/` — `hemorrhage_metrics_summary.csv` (raw), `.txt` + `.md` (readable reports), confusion matrix, error cases, plots.  
Verify_Vaskulär-only cases excluded from default metrics until label meaning is clarified.

### Phase 0 status

| Done | Not yet |
|------|---------|
| `ClinicalCase` model, case_id, grouping | Keyword prefilter |
| `case_builder`, inspection, reference analytics | Final validation (Verify_Vaskulär TBD) |
| Case-level LLM prototype (`run_case_pipeline`) | Guardrails |
| Prediction review + preliminary evaluation (`evaluate_predictions`) | Removal of delirium code |

### Inspect real server data (Phase 0b — no NLP)

```bash
python3 -m src.tasks.hemorrhage.inspect_data
```

Raw Excel (configurable, under `data/raw/`):

- `NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx` — clinical reports export
- `260507_CCM_DAVF.xlsx` — reference / manual labels (CCM DAVF)

Outputs: `data/inspection/` (schema, cases, merge, keywords, samples)

### Build cases from CSV (no NLP)

```bash
python3 -m src.tasks.hemorrhage.build_cases --input data/raw/reports.csv
```

Outputs:

- `outputs/prepared/cases/clinical_cases.csv` — **one row = one case**
- `outputs/prepared/cases/case_construction_report.txt`

### Environment

| Variable | Default | Role |
|----------|---------|------|
| `PROJECT_TASK` | `hemorrhage` | Task selector in `paths.py` |
| `FLAT_REPORTS_INPUT_PATH` | `data/raw/reports.csv` | Flat report input |
| `HEMORRHAGE_PREFILTER_MODE` | `disabled` | No delirium keyword auto-skip |
| `LLM_PROVIDER` | `usz_api` | LLM backend (`usz_api` or `ollama`) |
| `USZ_LLM_URL` | `http://localhost:8100/generate` | USZ LLM endpoint |
| `HEMORRHAGE_LLM_TIMEOUT_SECONDS` | `240` | Per-call LLM read timeout |
| `HEMORRHAGE_LLM_MAX_RETRIES` | `1` | Auto-retries on timeout/connection error (5 s wait) |

### Delirium pipeline (legacy, unchanged)

To run the original report-level delirium pipeline (unchanged code paths):

```bash
export PROJECT_TASK=delirium
python3 -m src.pipeline.run_pipeline
```

---

## Delirium legacy reference (report-centric)

The sections below describe the **copied delirium pipeline** still present for reference and regression.

## Project Purpose (delirium)
- Detect ICU delirium from clinical diagnosis text using a 3-agent pipeline.
- Compare model predictions against an operational structured baseline (ICD10 + ICDSC).
- Provide reproducible validation, evaluation, and exploratory analysis outputs.

## Core Architecture
- **Pre-LLM layer**: structured rule-based evidence extraction (`src/preprocessing/evidence_extraction.py`) — full `report_text` is scanned; only bounded, section-tagged snippets are assembled for the LLM.
- **Agent 1**: extraction (`src/agents/extraction.py`) — JSON signal buckets from the **evidence bundle** (not the full report).
- **Agent 2**: interpretation (rule/prompt; default prompt) (`src/agents/interpretation.py`, `src/agents/interpretation_llm.py`) — assigns **signalstaerke** (`niedrig` / `mittel` / `hoch`).
- **Agent 3**: classification (`src/agents/classification.py`) — **binary** `klasse` 0/1 from signal strength (`mittel`/`hoch` → 1, `niedrig` → 0).
- **Prediction unit**: one row per **report** in `Berichte.csv` (report-level). Patient-level validation uses `patient_reporttype_matrix.csv`.
- **Excluded from processing** (raw CSV unchanged): `bertyp == Dokumentationsblatt` — logged as `excluded_dokumentationsblatt_count`.

## Evidence extraction (scientific / scalability)
- **Binary output only**: `klasse` ∈ {0, 1}. There is **no** multiclass prediction head.
- **Signal strength** remains `niedrig` | `mittel` | `hoch` (interpretation only); mapping to `klasse` is unchanged.
- The **entire** stitched `report_text` is scanned with deterministic keyword groups:
  - **direct_delir**, **indirect_symptom**, **negation**, **prophylaxis_or_risk** (see `evidence_extraction.py`).
- **Negated** delirium phrases are **not** treated as positive evidence; **prophylaxis / screening / risk-only** mentions are **not** auto-positive for delirium (the LLM is instructed; final class still flows through signal strength).
- If **no** snippet qualifies for LLM review, the LLM is normally **skipped** (`no_evidence_prefilter_skip`, `klasse=0`). Optional **short-report fallback** (see env below) sends capped full text for brief `Verlaufseintrag` / `Verlegungsbericht` / `Austrittsbericht` without keyword hits.
- If actionable snippets exist, `llm_text_reduction_method=structured_evidence_extraction` and the LLM receives **`llm_report_text`**: labeled snippets — **not** the full chart.
- **LLM prompts** (`prompts/agent_extraction.txt`, `prompts/agent_interpretation.txt`) are **German**; JSON field names (`signalstaerke`, `alternative_erklaerung`, …) stay English for parsers.
- **Transparency**: describe this two-stage design (rules → LLM) in thesis/defense materials; CSV stores structured `evidence_snippets` (JSON list) plus boolean flags for audit.
- **Clinical guardrails** (`src/agents/clinical_guardrails.py`, after Agent 2): hard-excludes **no evidence**, **prophylaxis/conditional-only**, **negated delirium**, and **isolated weak indirect symptoms** (single agitation/vigilance/GCS-only). **Direct delir** and **delirium-compatible symptom clusters** (≥2 indirect dimensions or therapy+symptoms) may stay `klasse=1`; clusters with `alternative_erklaerung` are flagged for review. **Isolated indirect + dominant alternative** → `klasse=0`, `alternative_explanation_downgrade`.

### Environment (evidence + logging)
| Variable | Default | Role |
|----------|---------|------|
| `EVIDENCE_MAX_SNIPPETS` | 12 | Max structured snippets per patient. |
| `EVIDENCE_MAX_LLM_CHARS` | 8000 | Cap on assembled LLM evidence bundle size. |
| `EVIDENCE_WINDOW_SENTENCES` | 1 | Sentences before/after the hit sentence in each window. |
| `EVIDENCE_MAX_SNIPPET_CHARS` | 400 | Max characters per snippet `text` field. |
| `DEBUG_LLM_OUTPUT` | false | If true, print verbose per-agent debug (full previews, raw LLM). |
| `LLM_TEMPERATURE` | (provider default) | Recommended **0** for reproducible extraction/interpretation. |
| `LLM_TOP_P` | (provider default) | Recommended **1** with `LLM_TEMPERATURE=0`. |
| `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM` | false | If true, short reports without snippet hits still go to LLM (full capped text). |
| `SHORT_REPORT_CHAR_THRESHOLD` | 1000 | Max `original_report_text_length` for short-report fallback. |

## Pipeline stages
1. Prepare structured baseline (`src/pipeline/prepare_structured_data.py`)
2. Run text pipeline (`src/pipeline/run_pipeline.py`)
3. Compare predictions vs baseline (`src/pipeline/compare_reports_vs_baseline.py`)
4. Evaluate metrics/plots (`src/pipeline/evaluate_predictions.py`)
5. Validate input consistency (`src/validation/validate_inputs.py`)
6. Advanced exploration (`src/analysis/run_exploration.py`)
7. In-depth analysis (`src/analysis/run_analysis.py`)

## Data Sources (Current Default)
Centralized in `src/pipeline/paths.py`.

- `DATA_MODE = "real"` (default)
- **Final production raw inputs** (semicolon-separated under `data/raw/`):
  - `Berichte.csv` — primary text (`PatientID`, clinical fields → `report_text`)
  - `ICD.csv` — `PatientID; icd_hd; icd_code`
  - `ICDSC.csv` — `PatientID; ICDSC_Max` (patient-level maximum score)
- **No** `Diagnosenliste.csv` in the active pipeline (`LEGACY_DIAGNOSIS_INPUT_PATH` only for documentation).

Optional synthetic mode (`DATA_MODE = "synthetic"`):
  - `data/structured/raw/synthetic_icd10.csv`, `synthetic_icdsc.csv`, `synthetic_berichte.csv`
  - Legacy: `data/anonymized/beispiele/synthetic_diagnoses.csv` (INPUT_MODE=diagnosis only)

## Important Logic
- **Baseline construction** (`prepare_structured_data`): ICD + ICDSC only → `structured_baseline.csv`.
- **ICD-10 delirium** (`has_delir_icd10` / `baseline_icd10`): main diagnosis `icd_hd == 1` and codes **F05.0, F05.8, F05.9** only. **F05.1** (alcohol-related / Entzugsdelir) and other F05 subcodes are **excluded** from the intended cohort. Implementation: exact allowlist in `is_valid_delir_icd10_code()` — no `startswith("F05")` / no `INCLUDE_ALL_F05_PRESENTATION_MODE`.
- **ICDSC** (`max_icdsc`): from `ICDSC_Max`; thresholds `baseline_icdsc_ge_*`, `baseline_icdsc_0`, `baseline_icdsc_1_to_3`, `baseline_icdsc_ge_4_grouped`.
- **Primary validation baseline** `baseline_composite` — configurable in `src/pipeline/paths.py` as `BASELINE_COMPOSITE_MODE`:
  - **`OR`** (thesis): `(baseline_icdsc_ge_4 == 1) OR (baseline_icd10 == 1)` — broader/sensitive.
  - **`AND`** (temporary presentation): `(baseline_icdsc_ge_4 == 1) AND (baseline_icd10 == 1)` — high-confidence / «sichere Delirfälle».
  - Model-positive / AND-baseline-negative → interpret as **Delirkandidaten** (possible uncoded delir), not automatic false positives.
- **Legacy** multiclass `baseline_reference_class` may still be written; primary evaluation uses binary baselines including `baseline_composite`.
- **Deprecated:** `Diagnosenliste.csv` / `diagnosis_mapper` — not used in production. `Berichte.csv` columns `diag`, `epikrise`, `jetziges_leiden`, `prozedere` map to report sections `[Diagnosen]`, `[Epikrise]`, etc.
- **Exploration** (`run_exploration.py`): Berichte + ICD + ICDSC + structured baseline + predictions; no crash when legacy diagnosis path is absent.

## Single Source of Path Truth
- `src/pipeline/paths.py` is the central config.
- Do not hardcode paths elsewhere.
- New analysis/exploration output dirs are also defined there.

## Most Important Commands

### Full one-command run
```bash
./scripts/run_all.sh
```

### Preflight (recommended before sensitive runs)
```bash
./scripts/preflight_check.sh
```
### Pre-prompt (pilot / dev only — not thesis validation)

```bash
export MAX_REPORTS=60   # caps REPORT rows — do NOT use for final validation
export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
```

For the thesis validation path, use **[FINAL THESIS WORKFLOW (FULL RUN)](#final-thesis-workflow-full-run)** instead.

### Manual step-by-step (reference / partial runs)
```bash
python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.pipeline.compare_reports_vs_baseline
python3 -m src.pipeline.evaluate_predictions
python3 -m src.validation.validate_inputs
python3 -m src.analysis.run_exploration
python3 -m src.analysis.run_analysis
python3 -m src.analysis.run_validation_suite
python3 -m src.analysis.create_patient_reporttype_matrix
python3 -m src.analysis.export_patient_validation_cohort   # PRIMARY manual validation export
python3 -m src.analysis.export_manual_report_labels        # slim sheet for annotation
python3 -m src.analysis.freeze_validation_cohort         # lock 100-patient cohort (after full inference)
# Annotate manual_report_labels.csv (or frozen copy); then:
python3 -m src.analysis.evaluate_manual_validation         # prefers frozen_validation_cohort/ if present
python3 -m src.analysis.export_presentation_examples
```

## Validation architecture (PRIMARY)

| Unit | Definition |
|------|------------|
| Prediction | 1 report → 1 `model_report_prediction` (from `klasse`) |
| Validation cohort | 100 unique patients (`PATIENT_VALIDATION_N`); **all** evaluatable reports per patient (Berichte spine + predictions; includes prefilter-skipped rows) |
| Manual annotation | `manual_report_ground_truth` (0/1) per report |
| Patient manual GT | **Derived:** `derived_manual_patient_ground_truth` = max(report GT) per patient |
| ICDSC / ICD10 | **Reference signals only** — not absolute truth |

Legacy (deprecated): `export_manual_validation_sample`, `run_error_review_export` (`manual_label_0_1_2`).

---

## FINAL THESIS WORKFLOW (FULL RUN)

Copy-paste sequence for the **production thesis run**: full inference → validation cohort → freeze → manual annotation → evaluation.

Run all commands from the project root (`delirium_project/`). Ensure `data/raw/Berichte.csv`, `ICD.csv`, and `ICDSC.csv` are present and the USZ LLM service is reachable (see environment notes above).

### 1. Full dataset inference

For final thesis evaluation, the **full** report corpus must be processed **before** exporting the manual validation cohort.

**Why:** Validation is patient-level. Every selected patient must have a **complete report trajectory** (processed, prefilter-skipped, and guardrail-negative rows). The cohort export joins `Berichte.csv` with predictions; predictions must cover the full run first.

**Critical:** Do **not** set a numeric `MAX_REPORTS` for thesis validation. `MAX_REPORTS` limits **report rows**, not patients — a partial run under-represents patients and biases validation. Default in `paths.py` is already **no cap**; only `unset` if your shell still exports a limit from an earlier pilot.

```bash
cd delirium_project

unset MAX_REPORTS
# Optional explicit full run: export MAX_REPORTS=all

export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
export SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true
export SHORT_REPORT_CHAR_THRESHOLD=1000

# LLM (USZ) — adjust if your server differs
export LLM_PROVIDER=usz_api
export LLM_TEMPERATURE=0
export LLM_TOP_P=1

python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.analysis.run_validation_suite
```

| Step | Role |
|------|------|
| `prepare_structured_data` | Builds `structured_baseline.csv` (ICDSC + ICD10; F05.0/F05.8/F05.9 main diagnosis only). |
| `run_pipeline` | Report-level inference for **all** included Berichte rows (Dokumentationsblatt excluded). |
| `run_validation_suite` | Compare vs baseline, evaluation plots, patient matrix, **and** `export_patient_validation_cohort` (mutable copy). |

Optional after inference: `python3 -m src.validation.validate_inputs`

**Primary predictions file:** `outputs/predictions/agent1_agent2_agent3_results_prompt.csv`

### 2. Export validation cohort

Run **after** full inference. Re-export if predictions or baseline change.

```bash
export PATIENT_VALIDATION_N=100

python3 -m src.analysis.export_patient_validation_cohort
python3 -m src.analysis.export_manual_report_labels
```

| Output | Purpose |
|--------|---------|
| `outputs/analysis/manual_validation/patient_validation_cohort.csv` | Wide cohort (all reports for 100 patients; stable IDs). |
| `outputs/analysis/manual_validation/manual_report_labels.csv` | **Slim sheet for annotation** (copy before editing if not freezing immediately). |
| `outputs/analysis/manual_validation/patient_validation_cohort_report.txt` | Counts + processing summary (`status`, skipped vs LLM, etc.). |

**Included report types:** Verlaufseintrag, Verlegungsbericht, Austrittsbericht.

**Skipped reports are included** — prefilter skip (`no_evidence_prefilter_skip`) is still a model decision (`klasse=0`, `status=skipped`). Guardrail negatives after LLM are included too.

**Pilot (30 patients):** `export PATIENT_VALIDATION_N=30` with the same commands (only for dry runs, not thesis freeze).

### 3. Freeze validation dataset

Locks the cohort for manual review. **Do not re-freeze** after annotation starts unless you intentionally reset the study (requires `OVERWRITE_FROZEN_VALIDATION=true`).

```bash
python3 -m src.analysis.freeze_validation_cohort
```

**Fixed thesis dataset directory:**

`outputs/analysis/manual_validation/frozen_validation_cohort/`

| File | Role |
|------|------|
| `patient_validation_cohort_frozen.csv` | Full frozen cohort (reference). |
| `manual_report_labels_frozen.csv` | **Annotate this file** (or copy labels back into this path). |
| `frozen_cohort_metadata.json` | Timestamp, patient/report counts, checksums, source paths. |

### 4. Manual validation

**Primary annotation file:**

`outputs/analysis/manual_validation/frozen_validation_cohort/manual_report_labels_frozen.csv`

Annotate only:

- **`manual_report_ground_truth`** — required  
- `manual_comment` — optional  

**Allowed values:** `0` or `1`

| Value | Meaning |
|-------|---------|
| `1` | Clinically plausible delir documented in **this** report |
| `0` | No delir documented in **this** report |

**Do not** fill a manual patient-level label. Patient-level ground truth is derived automatically:

`derived_manual_patient_ground_truth = max(manual_report_ground_truth)` per `validation_patient_id`

Use columns `status`, `llm_called`, `skipped_reason` in the frozen full cohort for context (prefilter vs LLM-processed).

### 5. Final evaluation

```bash
python3 -m src.analysis.evaluate_manual_validation
```

Uses **frozen** cohort + frozen labels when `frozen_validation_cohort/` exists.

**Outputs:** `outputs/analysis/manual_validation/evaluation/` (metrics tables, confusion plots, `evaluation_report.txt`)

| Level | Comparison | Role |
|-------|------------|------|
| Report | `model_report_prediction` vs `manual_report_ground_truth` | Per-report agreement |
| **Patient (PRIMARY)** | `model_patient_positive` vs `derived_manual_patient_ground_truth` | **Main thesis evaluation** |
| Exploratory | ICDSC / ICD10 vs derived patient GT | Reference signals only — not absolute truth |

### 6. Important warnings

- **`MAX_REPORTS` limits reports, not patients** — never use it for the thesis validation run.
- **Never regenerate** the validation cohort after manual annotation has started (use the frozen copy).
- **Never mix** predictions from a new `run_pipeline` with an old frozen cohort — re-freeze only with a deliberate reset (`export OVERWRITE_FROZEN_VALIDATION=true`).
- **Always evaluate** using frozen files when they exist (`evaluate_manual_validation` prefers them automatically).
- **Skipped / prefilter-negative reports belong in the cohort** — excluding them inflates performance.
- **`Dokumentationsblatt` stays excluded** from processing and validation (raw CSV unchanged).
- **ICDSC / ICD10** in the cohort are reference signals, not manual ground truth.

---

## Validation outputs
- `outputs/analysis/patient_level/patient_reporttype_matrix.csv` — exploratory patient matrix.
- `outputs/analysis/manual_validation/patient_validation_cohort.csv` — **PRIMARY** export (`Patient_0001`, `Patient_0001_Report_0001`, …).
- `outputs/analysis/manual_validation/manual_report_labels.csv` — slim annotation sheet (merge by `validation_report_id` at evaluation).
- `outputs/analysis/manual_validation/frozen_validation_cohort/` — **fixed** cohort for thesis validation (`*_frozen.csv`, `frozen_cohort_metadata.json`). Re-freeze only with `OVERWRITE_FROZEN_VALIDATION=true`.
- `outputs/analysis/manual_validation/patient_validation_cohort_report.txt` — cohort summary + methodology.
- `outputs/analysis/manual_validation/evaluation/` — metrics after annotation (`evaluate_manual_validation`).
- Exploratory: `delir_probability_estimate` (0–100) in predictions CSV; not used for final `klasse`.
- `outputs/analysis/presentation_examples/` — CSV + Markdown examples for slides (excerpt → keywords → evidence → LLM → prediction).

## Output Structure
- `outputs/baseline/` → structured baseline tables
- `outputs/predictions/` → Agent 1/2/3 outputs
- `outputs/comparisons/` → merged prediction vs baseline
- `outputs/evaluation/` → multiclass metrics, confusion matrices, error exports, plots
- `outputs/validation/` → validation checks (`validation_results.csv`, `validation_summary.txt`)
- `outputs/analysis/`
  - `exploration/` → raw-data EDA tables/plots/report
  - `tables/`, `plots/`, `reports/` → in-depth analytical views

## Docker / Ubuntu Notes
- Dockerfile exists at `docker/Dockerfile`.
- Build:
```bash
docker build -f docker/Dockerfile -t delirium-pipeline .
```
- Example run:
```bash
docker run --rm -it \
  -v "$(pwd)/Data/Raw:/app/Data/Raw" \
  -v "$(pwd)/outputs:/app/outputs" \
  delirium-pipeline \
  python -m src.pipeline.prepare_structured_data
```

## Sensitive Data Safety
- `data/` and `Data/` are gitignored.
- Keep real ICU data only local/server-side.
- Avoid committing raw input files.

## Current Known Practical State
- Code and tests are passing in the current local setup.
- Placeholder `Data/Raw` files exist for dry-run wiring; replace with real files on Ubuntu.
- If ICD files are empty, pipeline runs but baseline/evaluation quality is limited (expected).

## Quick Troubleshooting
- No reports processed (`Anzahl Berichte: 0`):
  - Check `data/raw/Berichte.csv` exists and has rows with `PatientID`.
- Baseline empty or wrong joins:
  - Re-run `python -m src.pipeline.prepare_structured_data`.
  - Check `data/raw/ICD.csv` (`PatientID; icd_hd; icd_code`) and `data/raw/ICDSC.csv` (`PatientID; ICDSC_Max`).
- Format issues:
  - Reader supports `.csv`, `.xlsx`, `.xls` via `src/pipeline/tabular_io.py`.
