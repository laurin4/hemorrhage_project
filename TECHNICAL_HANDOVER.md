# Technical Handover & Architecture Analysis

**Document purpose:** Deep engineering handover from the delirium ICU pipeline (`delirium_project`) to enable a clean, consistent build of the hemorrhage classification subproject (`hemorrhage_project`).

**Source of truth analyzed:** `/delirium_project` (production thesis codebase)  
**Target repo:** `/hemorrhage_project`  
**Last updated:** 2026-05-23 (Phase 0 case-centric refactor)  
**Audience:** Developers implementing hemorrhage transfer — not end users.

---

## Two-level classification target (supervisor clarification)

The target is no longer purely binary. The pipeline now produces a two-level label.

- **Level 1 (always):** `klasse=0 → nicht_hämorrhagisch`, `klasse=1 → hämorrhagisch`.
- **Level 2 (only if `klasse=1`):** `haemorrhage_subtype` ∈ {`akut`, `nicht_akut`, `historisch`} (mandatory when hemorrhagic).
  - `nicht_hämorrhagisch` → `haemorrhage_subtype = null`.
  - hämorrhagisch with missing/unrecognized subtype → parser sets `haemorrhage_subtype = "unbekannt"` and appends an uncertainty reason (parse does **not** fail). The model is instructed not to emit `unbekannt`.
- **Historical hemorrhage is still hemorrhage:** a past/remote bleed is `klasse=1` + `subtype="historisch"`, NEVER `klasse=0`. (`akut` = current acute event; `nicht_akut` = current non-acute finding; `historisch` = past/old, even if acute at the time.)
- **`Verify_Vaskulär` is metadata only** — never a class. It must not influence classification, and `verify_only` reference cases stay excluded from binary metrics (unless `--include-verify-as-negative`).

### Final clinical working definition + inference cohort

**Binary (Stage 1):** `klasse=1` needs explicit/clinically relevant hemorrhage evidence (Blutung, Einblutung/eingeblutet, geblutet, hämorrhagisch/hemorrhagic, Hämatom/Hematom, Hämatomevakuation, clear bleeding treatment context). Insufficient alone for klasse=1: Kavernom/CCM, DAVF/AVM/vascular lesion, epilepsy, resection/operation, vascular verification, lesion diagnosis without bleeding wording. Historical hemorrhage stays klasse=1.

**Subtype (Stage 2, only if klasse=1):** `historisch` = only a previous/historical event; `nicht_akut` = relevant in the current case but not acute; `akut` = acute/fresh/subacute hemorrhage or acute hemorrhage-related treatment (incl. hematoma evacuation).

**Inference cohort filter** (`inference/runner.py::_filter_to_cohort` + `io/reference_lookup.py::reference_binary_status`): default processes only `hemorrhagic`/`non_hemorrhagic`; excludes `verify_only`/`unknown`/`inconsistent`. CLI flags `--all-cases`, `--include-verify-only`; `--case-id` bypasses; missing reference → filter skipped (process all + warning). `derive_reference_status` in `export/prediction_review.py` now delegates to `reference_binary_status` (single source of truth). Startup + summary print `cohort_mode` and `excluded_by_status`.

### Two-stage hierarchical inference (current pipeline path)

To reduce per-call token generation and `ReadTimeout`s, the runner issues **two sequential LLM calls** per case instead of one combined call:

- **Stage 1 — binary** (`prompts/hemorrhage_binary_classification.txt`, builders `build_binary_messages` / `build_binary_user_prompt`, parser `parse_binary_response`): decides only `klasse` 0/1; compact JSON `{klasse, label, sicherheit, kurzbegruendung}` (no evidence list; `parse_binary_response` reads `kurzbegruendung` as `begruendung`); leaves `haemorrhage_subtype = None`.
- **Stage 2 — subtype** (`prompts/hemorrhage_subtype_classification.txt`, builders `build_subtype_messages` / `build_subtype_user_prompt`, parser `parse_subtype_response` → `SubtypeParseResult`): runs **only when `klasse=1`**; decides only the subtype, assuming hemorrhage exists.
- Merge: `runner._merge_subtype` combines Stage 1 (klasse/label/evidence/reasoning) with Stage 2 (subtype/evidence/reasoning) into one row. CSV schema is unchanged.
- Failure semantics: Stage 1 LLM error → `status=llm_failed`; Stage 1 parse error → `status=parse_failed` (`parse_failures` CSV uses the binary raw). Stage 2 LLM/parse failure does **not** drop the case → it stays `hämorrhagisch` with `haemorrhage_subtype="unbekannt"` and `subtype_stage_status` ∈ {`llm_failed`, `subtype_unknown`}.
- New columns: `binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length`. `raw_response_length` = len(binary raw) + len(subtype raw); `raw_llm_response` holds both stages separated by `--- SUBTYPE STAGE ---`.
- `parse.py` shares a `_parse_binary_core` helper between the combined and binary parsers, so `parse_hemorrhage_response` (single-call) behaves exactly as before and remains for tests/legacy.

Implementation touch points:

- Prompts: two-stage `prompts/hemorrhage_binary_classification.txt` + `prompts/hemorrhage_subtype_classification.txt`; combined `prompts/hemorrhage_case_classification.txt` (single-call/legacy). Fallbacks in `inference/prompt.py`.
- Parser/schema: `inference/parse.py` (`normalize_haemorrhage_subtype`, `_resolve_haemorrhage_subtype`, `VALID_HAEMORRHAGE_SUBTYPES`, `SUBTYPE_UNKNOWN`). Subtype normalization maps `acute→akut`, `history/historical→historisch`, `non_acute/chronisch/chronic/nicht-akut→nicht_akut`.
- Prediction CSV: `haemorrhage_subtype` + two-stage columns (`binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length`) in `inference/runner.py`.
- Review exports: `predicted_haemorrhage_subtype` + `reference_haemorrhage_subtype` (empty placeholder) in prediction/confusion/FN/FP CSVs (`export/prediction_review.py`).
- Reference: `io/reference_lookup.py` adds `reference_haemorrhage_subtype = ""` (no invented subtype).
- Evaluation: `evaluation/runner.py` adds `compute_subtype_counts`, subtype distribution + subtype-by-reference CSVs, two subtype plots; binary metric computation is unchanged. Subtype accuracy is **not** computed (descriptive only).
- Reports: `evaluation/report_format.py` adds a "Hemorrhage subtype analysis" section to the TXT and MD reports.

---

## Phase 0 — Case-centric architecture (IMPLEMENTED)

### Methodological shift

| Delirium (legacy) | Hemorrhage (target) |
|-------------------|---------------------|
| 1 report → 1 prediction | **1 case → 1 prediction** |
| Patient = max(report preds) | Case = grouped reports |
| `PatientID` + `bertyp` | `(excel_pid, excel_opdat, opber_fallnr)` |

### Clinical case definition

```text
Case = (excel_pid, excel_opdat, opber_fallnr)
```

Expected report typus per case (any subset — **incomplete cases are normal**):

| Code | Label |
|------|-------|
| `01` | Operationsbericht |
| `02` | Eintrittsbericht |
| `03` | Austrittsbericht |

The constructor **never requires all three**. It records `available_report_types` and `missing_report_types` explicitly.

### New architecture tree (Phase 0)

```text
src/
  core/                          # Task-agnostic framework (new)
    case/                        # ClinicalCase, CaseKey, case_id
    legacy/report_centric.py     # Registry of report-level assumptions
    io/                          # Re-export tabular I/O
    identity/                    # Re-export report-row IDs (delirium merge)
    llm/                         # Re-export LLM transport
  tasks/
    hemorrhage/                  # Case-centric hemorrhage (active)
      constants.py
      inference_policy.py        # HEMORRHAGE_PREFILTER_MODE=disabled (default)
      preprocessing/case_builder.py
      export/case_export_schema.py
      build_cases.py             # CLI
    delirium/                    # Boundaries doc only — code still in src/pipeline
      BOUNDARIES.md
  pipeline/                      # Legacy delirium orchestration (unchanged behavior)
  agents/                        # Delirium agents (isolated, not deleted)
  preprocessing/               # Delirium evidence + berichte_mapper
  analysis/                      # Report/patient validation (delirium)
```

### Phase 1 — Case-level LLM inference (IMPLEMENTED)

```bash
python3 -m src.tasks.hemorrhage.run_case_pipeline --dry-run --limit 5
python3 -m src.tasks.hemorrhage.run_case_pipeline --limit 20
python3 -m src.tasks.hemorrhage.run_case_pipeline
```

- Entry: `src/tasks/hemorrhage/run_case_pipeline.py`; orchestration in `inference/runner.py`; LLM transport in `inference/llm_client.py`.
- Prompts: two-stage `prompts/hemorrhage_binary_classification.txt` (Stage 1) + `prompts/hemorrhage_subtype_classification.txt` (Stage 2); historical hemorrhage = `klasse=1` + `subtype=historisch`.
- Output: `data/outputs/hemorrhage_case_predictions.csv` (incl. `haemorrhage_subtype`, `binary_stage_status`, `subtype_stage_status`, `binary_prompt_length`, `subtype_prompt_length` + debug columns `prompt_length_chars`, `structured_case_text_length`, `raw_response_length`).
- One case = one prediction row. **Two LLM calls** for hemorrhagic cases (binary + subtype), **one** for non-hemorrhagic (subtype skipped). No keyword prefilter; delirium `run_pipeline.py` unchanged.
- **Robustness:** per-call timeout `HEMORRHAGE_LLM_TIMEOUT_SECONDS` (default 240), `HEMORRHAGE_LLM_MAX_RETRIES` (default 1, retries on `ReadTimeout`/`Timeout`/`ConnectionError`). On failure → `status=llm_failed` and the run continues. Predictions are written **incrementally** (partial-save safe).

### Phase 1b — Preliminary evaluation (IMPLEMENTED)

```bash
python3 -m src.tasks.hemorrhage.build_prediction_review
python3 -m src.tasks.hemorrhage.evaluate_predictions
python3 -m src.tasks.hemorrhage.evaluate_predictions --include-verify-as-negative
```

- Entry: `src/tasks/hemorrhage/evaluate_predictions.py`
- Core logic: `src/tasks/hemorrhage/evaluation/runner.py` (readable reports in `evaluation/report_format.py`; `export/evaluate_predictions.py` is a backward-compat shim).
- Inputs: `data/outputs/hemorrhage_prediction_review.csv`, `hemorrhage_confusion_review.csv`
- Outputs: `data/evaluation/` (metrics `.csv`/`.txt`/`.md`, confusion matrix, error cases, subtype tables `hemorrhage_subtype_distribution.csv` + `hemorrhage_subtype_by_reference_status.csv`, `plots/*.png`)
- **Binary evaluation** = hämorrhagisch vs nicht_hämorrhagisch (historical counts as positive); subtype is **descriptive only** and never affects TP/TN/FP/FN. Verify_Vaskulär-only cases excluded from default metrics.
- Optional sensitivity: `--include-verify-as-negative` treats verify_only as non_hemorrhagic.

See **HANDOVER_SUMMARY.md** → “How to run on the server (copy-paste, full workflow)” for the complete copy-paste workflow incl. environment variables.

### Phase 0b — Real Excel inspection (IMPLEMENTED)

```bash
python3 -m src.tasks.hemorrhage.inspect_data
```

**Configured paths** (`src/tasks/hemorrhage/config.py` + env):

| File | Default path |
|------|----------------|
| Reports (clinical export) | `data/raw/NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx` |
| Reference / labels (CCM DAVF) | `data/raw/260507_CCM_DAVF.xlsx` |

Alternate for reference (logged, not renamed): `260507 CCM DAVF.xlsx`

**Outputs:** `data/inspection/` — `raw_schema_summary.csv`, `case_summary.csv`, `merge_validation.csv`, `incomplete_cases.csv`, `structured_case_samples.csv`, etc.

### Phase 0 commands

```bash
# Build cases from flat report CSV (no NLP)
python3 -m src.tasks.hemorrhage.build_cases --input data/raw/reports.csv

# Outputs:
#   outputs/prepared/cases/clinical_cases.csv      (one row = one case)
#   outputs/prepared/cases/case_construction_report.txt
```

**Input default:** `data/raw/reports.csv` (`FLAT_REPORTS_INPUT_PATH` env to override).

### Incomplete case handling (internal)

1. Flat rows grouped by `(excel_pid, excel_opdat, opber_fallnr)` — group always kept.
2. Each typus slot filled at most once (duplicate typus → first wins, logged).
3. Missing typus codes listed in `missing_report_types` (e.g. `02|03`).
4. Zero-report cases still exported (empty text rows logged as anomalies).
5. Missing key components use `__MISSING__` token — case preserved, flagged.
6. Prefilter: `HEMORRHAGE_PREFILTER_MODE=disabled` (default) — delirium auto-skip **not** used for hemorrhage paths.

### What Phase 0 did NOT change

- `src.pipeline.run_pipeline` — still report-centric delirium (unchanged).
- Delirium prompts, guardrails, ICD/ICDSC baseline — isolated under `src/tasks/delirium/BOUNDARIES.md`.
- No hemorrhage NLP, keywords, labels, or evaluation yet.

### Report-centric modules (isolated, not removed)

See `src/core/legacy/report_centric.py` for the full registry (`run_pipeline`, validation cohort export, etc.).

---

## Table of contents

1. [High-level architecture](#1-high-level-architecture)
2. [Project structure](#2-project-structure)
3. [Data model](#3-data-model)
4. [Identifier & merge logic](#4-identifier--merge-logic)
5. [Pipeline modules](#5-pipeline-modules)
6. [LLM system](#6-llm-system)
7. [Configuration system](#7-configuration-system)
8. [Validation & evaluation](#8-validation--evaluation)
9. [Current technical debt](#9-current-technical-debt)
10. [Reusability analysis](#10-reusability-analysis)
11. [Execution workflow](#11-execution-workflow)
12. [Exports & outputs](#12-exports--outputs)
13. [Methodological assumptions (delirium-embedded)](#13-methodological-assumptions-delirium-embedded)
14. [Hemorrhage transfer analysis](#14-hemorrhage-transfer-analysis)
15. [Current project state & risks](#15-current-project-state--risks)

---

## 1. High-level architecture

### 1.1 System purpose (delirium)

Binary detection of **documented ICU delirium** from German clinical report text (`Berichte.csv`), using a **hybrid rule + LLM** pipeline. Operational structured baselines (ICD-10 F05.x + ICDSC) exist for exploratory comparison but **manual report/patient labels** are the primary thesis validation ground truth.

### 1.2 Prediction unit vs validation unit

| Concept | Unit | Notes |
|---------|------|-------|
| **Inference** | 1 row in `Berichte.csv` → 1 prediction | Report-level `klasse` ∈ {0,1} |
| **Manual annotation** | Per report (`manual_report_ground_truth`) | Values 0/1 only |
| **Primary thesis metric** | Per patient | `model_patient_positive = max(report predictions)` vs `derived_manual_patient_ground_truth = max(report GT)` |

### 1.3 End-to-end execution flow (actual)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW INPUTS (data/raw/, semicolon CSV, gitignored)                           │
│   Berichte.csv  │  ICD.csv  │  ICDSC.csv                                    │
└────────┬────────────────────────────┬───────────────────────────────────────┘
         │                            │
         │                            ▼
         │              prepare_structured_data.py
         │              → outputs/baseline/structured_baseline.csv
         │                 (patient-level ICD10 + ICDSC flags)
         │
         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ run_pipeline.py  (INPUT_MODE=berichte, INTERPRETATION_MODE=prompt)         │
│                                                                            │
│  berichte_mapper.build_report_level_berichte_records()                     │
│    → assign source_report_row_id BEFORE bertyp filter                      │
│    → exclude Dokumentationsblatt                                           │
│    → stitch report_text from [Diagnosen]/[Jetziges Leiden]/[Epikrise]/...  │
│                                                                            │
│  FOR EACH report:                                                          │
│    1. evidence_extraction.extract_delirium_evidence(full_text)             │
│       → bounded snippets + llm_report_text (NOT full chart)                │
│    2. IF no snippets AND NOT short-report fallback → SKIP LLM              │
│       → klasse=0, status=skipped, guardrails no_evidence_prefilter_skip    │
│    3. ELSE Agent 1 extraction (LLM JSON signal buckets)                    │
│    4. Agent 2 interpretation_llm (signalstaerke niedrig|mittel|hoch)      │
│    5. Agent 3 classification (preliminary klasse from signal)            │
│    6. clinical_guardrails (final klasse, decision_rule_applied)            │
│    7. delirium_probability_estimate (exploratory 0–100, NOT final class)   │
│                                                                            │
│  → outputs/predictions/agent1_agent2_agent3_results_prompt.csv             │
└────────┬───────────────────────────────────────────────────────────────────┘
         │
         ├─► compare_reports_vs_baseline.py
         │     JOIN predictions ↔ baseline ON PatientenID (patient-level baseline
         │     duplicated onto every report row)
         │     → outputs/comparisons/report_vs_baseline_comparison.csv
         │
         ├─► evaluate_predictions.py
         │     Binary metrics vs each baseline column + plots
         │     → outputs/evaluation/binary_baselines/
         │
         ├─► validate_inputs.py
         │     → outputs/validation/validation_results.csv
         │
         └─► analysis/* (exploration, validation cohort, manual eval)
               export_patient_validation_cohort → freeze → annotate → evaluate_manual_validation
```

### 1.4 Layer responsibilities

| Layer | Role | Delirium-specific? |
|-------|------|-------------------|
| **Berichte ingestion** | CSV load, section stitching, row IDs | Partially (section set) |
| **Rule evidence** | Keyword scan, snippet windows, caps | **Yes** (delirium lexicon) |
| **LLM agents** | Structured extraction + interpretation | **Yes** (prompts, JSON schema) |
| **Guardrails** | Deterministic post-LLM overrides | **Yes** (delirium clinical rules) |
| **Structured baseline** | ICD-10 + ICDSC patient flags | **Yes** (F05, ICDSC thresholds) |
| **Manual validation** | Cohort export, freeze, metrics | Mostly **generic** pattern |

### 1.5 Key architectural decision: two-stage evidence → LLM

The **full** `report_text` is always scanned deterministically. The LLM normally receives only a **bounded evidence bundle** (`llm_report_text`), not the full chart. This was a deliberate scalability and hallucination-control choice.

**Exception:** `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true` sends capped full text for short `Verlaufseintrag` / `Verlegungsbericht` / `Austrittsbericht` without keyword hits.

---

## 2. Project structure

**Root:** `hemorrhage_project/` (mirror of `delirium_project/` at time of handover)

There is **no** separate `config/` package. Configuration lives in `src/pipeline/paths.py` and `src/models/model_config.py`.

### 2.1 `src/pipeline/` — orchestration & I/O paths

| Module | Purpose | Inputs | Outputs | Reusable? |
|--------|---------|--------|---------|-----------|
| `paths.py` | Single source of truth for all paths, `DATA_MODE`, `MAX_REPORTS`, `BASELINE_COMPOSITE_MODE` | Env `MAX_REPORTS` | Path constants | **Yes** (rename dirs/constants) |
| `run_pipeline.py` | Main inference loop | Berichte records | Predictions CSV (+ optional SQLite) | **Pattern yes**, logic no |
| `prepare_structured_data.py` | Patient baseline from ICD+ICDSC | `ICD.csv`, `ICDSC.csv` | `structured_baseline.csv` | **No** (hemorrhage needs different baseline) |
| `compare_reports_vs_baseline.py` | Report preds × patient baseline | Predictions, baseline | Comparison CSV | **Pattern yes** |
| `evaluate_predictions.py` | Metrics & confusion plots | Comparison CSV | Evaluation tables/plots | **Mostly yes** |
| `baseline_composite.py` | OR/AND composite baseline | ICD10+ICDSC flags | `baseline_composite` | **No** |
| `schema_normalize.py` | Column aliases, F05 allowlist, schema asserts | Raw ICD/ICDSC | Normalized frames | **Partial** (ID norm yes, F05 no) |
| `validation_cohort_filter.py` | `VALIDATION_COHORT_ONLY` subset run | Frozen cohort CSV | Filtered report list | **Yes** |
| `tabular_io.py` | CSV/XLSX reader | Paths | DataFrames | **Yes** |
| `sqlite_logging.py` | Optional per-row DB log | Prediction rows | SQLite DB | **Yes** |

### 2.2 `src/preprocessing/` — text & evidence before LLM

| Module | Purpose | Inputs | Outputs | Reusable? |
|--------|---------|--------|---------|-----------|
| `berichte_mapper.py` | Load Berichte, report-level records, section stitch | `Berichte.csv` | List[dict] per report | **Mostly yes** (section map may change) |
| `report_identity.py` | `source_report_row_id`, merge key selection | DataFrames | ID columns | **Yes** |
| `berichte_filters.py` | Exclude Dokumentationsblatt; matrix report types | DataFrame | Filtered DF | **Partial** (report-type set task-specific) |
| `evidence_extraction.py` | Rule-based snippet assembly | `report_text` | Evidence dict + `llm_report_text` | **No** (rewrite for hemorrhage) |
| `delirium_hint_keywords.py` | Central keyword list for analyses | — | Keywords | **No** |
| `report_text_llm_reduction.py` | Legacy keyword reduction | Text | Reduced text | **Deprecated** for main path |
| `diagnosis_mapper.py` | Legacy Diagnosenliste input | Synthetic CSV | Patient-level text | **Legacy only** |

### 2.3 `src/agents/` — LLM agents & guardrails

| Module | Purpose | Reusable? |
|--------|---------|-----------|
| `extraction.py` | Agent 1: load prompt, call LLM, parse JSON buckets | **Pattern yes**, content no |
| `interpretation_llm.py` | Agent 2 (default): LLM interpretation | **Pattern yes**, content no |
| `interpretation.py` | Agent 2 rule-based (not default) | Delirium-specific |
| `classification.py` | Agent 3: signal → preliminary klasse | **Pattern yes** |
| `clinical_guardrails.py` | Post-LLM deterministic overrides | **No** (delirium rules) |
| `delirium_probability.py` | Exploratory 0–100 score | **No** |
| `compare_rule_vs_prompt.py` | Dev comparison utility | Optional |

### 2.4 `src/models/` — LLM transport

| Module | Purpose | Reusable? |
|--------|---------|-----------|
| `model_config.py` | Provider URLs, temperature, timeouts | **Yes** |
| `llm_interface.py` | `call_llm`, USZ generate API, Ollama, JSON strip | **Yes** |
| `json_parsing.py` | Fence strip, parse, debug | **Yes** |
| `llm_debug.py` | Failed call dumps | **Yes** |

### 2.5 `src/analysis/` — exploration & validation

| Module | Purpose | Reusable? |
|--------|---------|-----------|
| `export_patient_validation_cohort.py` | **PRIMARY** 100-patient cohort export | **Yes** (column names need generalization) |
| `validation_cohort_reports.py` | Berichte spine + LEFT merge predictions | **Yes** |
| `validation_ids.py` | `Patient_0001_Report_0002` IDs | **Yes** |
| `freeze_validation_cohort.py` | Lock cohort + checksums | **Yes** |
| `frozen_validation_cohort.py` | Frozen path guards | **Yes** |
| `export_manual_report_labels.py` | Slim annotation sheet | **Yes** |
| `manual_report_labels.py` | Merge labels at evaluation | **Yes** |
| `evaluate_manual_validation.py` | CLI for manual metrics | **Yes** |
| `manual_validation_eval.py` | Metrics, patient max aggregation | **Yes** |
| `patient_reporttype_matrix.py` | Patient × bertyp matrix | **Partial** (bertyp axis may differ) |
| `run_validation_suite.py` | compare → eval → matrix → export | **Partial** |
| `run_field_delirium_analysis.py` | Field keywords vs baseline | **No** |
| `run_exploration.py`, `run_analysis.py` | EDA orchestrators | **Partial** |
| Other `run_*` analysis scripts | Keyword/overlap/coverage studies | Mostly delirium-specific |

### 2.6 `src/validation/`

| Module | Purpose | Reusable? |
|--------|---------|-----------|
| `validate_inputs.py` | Consistency checks on raw + baseline | **Partial** (checks are delirium-oriented) |

### 2.7 `prompts/`

| File | Purpose |
|------|---------|
| `agent_extraction.txt` | Agent 1 system prompt (German, delirium JSON schema) |
| `agent_interpretation.txt` | Agent 2 system prompt (`signalstaerke`, alternatives) |

### 2.8 `scripts/`

| Script | Purpose |
|--------|---------|
| `run_all.sh` | Full pipeline chain |
| `preflight_check.sh` | Smoke test on real CSVs |
| `test_usz_llm_api.py` | LLM connectivity |
| `generate_synthetic_*.py` | Offline test data |
| `check_no_sensitive_files.py` | Secret path guard |
| `build_wheelhouse_linux.sh` | Offline Linux wheels |

### 2.9 `tests/`

Comprehensive pytest suite (~40 files) covering evidence extraction, guardrails, validation cohort merge, frozen cohort, baseline modes, pipeline helpers. **High reuse value** for regression after generalization.

---

## 3. Data model

### 3.1 Raw inputs (`data/raw/`, semicolon-separated, not in git)

#### `Berichte.csv`

| Column | Role |
|--------|------|
| `PatientID` | **Required** patient key |
| `berdat` | Report date (sorting, merge fallback) |
| `bertyp` | Report type (filtered for validation) |
| `bername` / `bericht` | Report name → `pipeline_bericht` id |
| `diag`, `epikrise`, `jetziges_leiden`, `prozedere` | Stitched into labeled `report_text` |

**Stitched format:**
```text
[Diagnosen]
<diag text>

[Jetziges Leiden]
...

[Epikrise]
...

[Prozedere]
...
```

**Excluded from pipeline:** `bertyp == Dokumentationsblatt`  
**Validation/matrix types:** `Verlaufseintrag`, `Verlegungsbericht`, `Austrittsbericht`

#### `ICD.csv`
`PatientID; icd_hd; icd_code` → patient-level delirium ICD flags (F05.0, F05.8, F05.9 main diagnosis only)

#### `ICDSC.csv`
`PatientID; ICDSC_Max` → patient-level max screening score

> **Hemorrhage note:** Hemorrhage project may use different structured inputs (imaging codes, OP findings) or no ICDSC at all. Do not assume these files exist.

### 3.2 Intermediate: report record (in-memory dict)

```python
{
  "PatientenID": str,
  "bericht": str,           # pipeline id (bername or synthetic)
  "bertyp": str,
  "berdat": str,
  "source_report_row_id": str,  # berichte_row_<index>
  "report_text": str,       # stitched sections only
}
```

### 3.3 Predictions CSV (`agent1_agent2_agent3_results_prompt.csv`)

**Grain:** one row per processed report.

**Identity columns:** `PatientenID`, `bericht`, `bertyp`, `berdat`, `source_report_row_id`

**Evidence metadata:** `original_report_text_length`, `llm_report_text_length`, `llm_text_reduction_method`, `delir_keyword_hits_count`, `has_*_delir_evidence`, `evidence_snippets` (JSON list)

**`evidence_snippets` JSON element:**
```json
{
  "section": "diag",
  "keyword": "delir",
  "evidence_type": "direct_delir",
  "priority": 1,
  "text": "…windowed snippet…"
}
```

**Processing:** `status` ∈ {`skipped`, `processed`, `failed`}, `llm_called`, `skipped_reason`, `llm_skipped_by_prefilter`

**Agent outputs:** `delir_signale`, `signalstaerke`, `kontext`, `alternative_erklaerung`, `begruendung`

**Final:** `klasse` (0/1), `klassifikation`, `klassifikation_begruendung`, `decision_rule_applied`, `manual_review_candidate`

### 3.4 Structured baseline (`structured_baseline.csv`)

**Grain:** one row per patient.

| Column | Meaning |
|--------|---------|
| `PatientenID` | Patient key |
| `has_delir_icd10`, `delir_codes` | ICD-10 delirium (main dx, F05.0/8/9 only) |
| `max_icdsc` | Max ICDSC score |
| `baseline_icd10` | Binary |
| `baseline_icdsc_ge_1` … `baseline_icdsc_ge_5` | Threshold flags |
| `baseline_icdsc_0`, `baseline_icdsc_1_to_3`, `baseline_icdsc_ge_4_grouped` | Groups |
| `baseline_composite` | OR or AND of ICDSC≥4 and ICD10 |
| `baseline_reference_class` | Legacy 0/1/2 multiclass |
| `baseline_delir_reference` | Legacy binary |

### 3.5 Manual validation cohort (`patient_validation_cohort.csv`)

**Grain:** one row per **included** report for each selected patient (typically 100 patients).

**Stable annotation IDs:**
- `validation_patient_id` → `Patient_0001`
- `validation_report_id` → `Patient_0001_Report_0002`
- `report_nr_within_patient` → chronological index

**Model fields:** `model_report_prediction`, `model_patient_positive`, `signalstaerke`, `status`, `llm_called`, `skipped_reason`, …

**Manual fields:** `manual_report_ground_truth` (0/1), optional confidence/flags/comments

**Derived (do not annotate manually):** `derived_manual_patient_ground_truth`, `n_positive_reports_manual`

**Reference only:** `baseline_icd10`, `baseline_icdsc_ge_4`, `baseline_composite_or`, `baseline_composite_and`

### 3.6 Frozen validation (`frozen_validation_cohort/`)

| File | Role |
|------|------|
| `patient_validation_cohort_frozen.csv` | Immutable full cohort |
| `manual_report_labels_frozen.csv` | Annotate this file |
| `frozen_cohort_metadata.json` | Checksums, counts, source paths, timestamp |

---

## 4. Identifier & merge logic

### 4.1 Identifier hierarchy

```text
source_report_row_id     PRIMARY merge key (berichte_row_<pandas_index>)
                         Assigned on FULL loaded CSV BEFORE bertyp filter.
                         Survives Dokumentationsblatt exclusion for remaining rows.

pipeline_bericht         Same as run_pipeline "bericht" field:
                         bername if non-empty, else "{bertyp}_{PatientenID}_{index}"

PatientenID / PatientID  Patient key (normalized to PatientenID in pipeline)

validation_patient_id    Opaque cohort label Patient_NNNN (not equal to PatientenID)

validation_report_id     Patient_NNNN_Report_MMMM within cohort export
```

### 4.2 Merge key selection (`choose_prediction_merge_keys`)

**Priority:**
1. `source_report_row_id` — if present in spine and predictions with coverage
2. `(PatientenID, pipeline_bericht)` — predictions store pipeline id in `bericht` column
3. **Fallback:** `(PatientenID, bertyp, berdat, bericht)` — legacy exports

**Validation cohort merge** (`build_complete_validation_reports_frame`):
- Raw Berichte spine is **authoritative** (all included rows per patient)
- Predictions are **LEFT JOINed** (`validate="m:1"`)
- Row count must equal spine count (`assert_spine_row_count_preserved`)
- Unmatched spine rows → `status=missing_prediction`, implicit `klasse=0` fill

### 4.3 Baseline comparison join

`compare_reports_vs_baseline.py` joins on **`PatientenID` only**.

**Consequence:** Every report for a patient inherits the same patient-level baseline flags. A report-level negative can coexist with patient-level baseline positive — export warns via `report_patient_level_warning`.

### 4.4 Patient-level aggregation (model & manual)

| Aggregation | Formula | Used for |
|-------------|---------|----------|
| Model patient positive | `max(model_report_prediction)` per patient | Primary thesis metric |
| Manual patient GT | `max(manual_report_ground_truth)` per patient | Derived — never hand-labeled |
| Positive report count | `sum(manual_report_ground_truth==1)` | Descriptive |

### 4.5 Historical bugs & fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Validation cohort missing prefilter-skipped reports | Early exports deduplicated or used predictions-only spine | **Berichte spine authoritative**; include `status=skipped` rows |
| Row count mismatch patient export | INNER join or prediction-only export | LEFT merge + `assert_spine_row_count_preserved` |
| Predictions not matching after pipeline rerun | Missing `source_report_row_id` in old CSV | Assign IDs before filter; enrich legacy preds from spine |
| `MAX_REPORTS` broke validation | Caps **reports** not patients | Documented: never use for thesis; `unset MAX_REPORTS` |
| Mixed frozen cohort + new predictions | Re-export without re-freeze | `evaluate_manual_validation` prefers frozen; `OVERWRITE_FROZEN_VALIDATION` guard |
| Implicit negative for missing preds | Unmatched spine rows filled `klasse=0` | Explicit `missing_prediction` status — still counts as model decision |

### 4.6 Remaining risks

- **Fallback merge** on `(bertyp, berdat, bericht)` can collide if dates/names duplicate
- **`pipeline_bericht` depends on pandas index** — reordering CSV changes synthetic ids
- **Patient-level baseline join** inflates report-level baseline agreement metrics
- **No automatic LLM retry** — transient failures → `status=failed`, `klasse=0`
- **hemorrhage fork** still has delirium filenames/prompts — easy to run wrong experiment

---

## 5. Pipeline modules

### 5.1 Keyword / evidence extraction (`evidence_extraction.py`)

**Purpose:** Scan full `report_text` with deterministic keyword groups; build ranked, capped snippets for LLM.

**Keyword groups:**
- `direct_delir`, `indirect_symptom`, `negation`, `prophylaxis_or_risk`
- Section-aware via `[Diagnosen]` etc. markers

**Outputs:** `llm_report_text`, boolean flags, `evidence_snippets` JSON, `llm_text_reduction_method`

| Question | Answer |
|----------|--------|
| Reusable? | **No** — rewrite lexicon & evidence types for hemorrhage |
| Hemorrhage-compatible? | Pattern only (snippet windows, caps) |
| Refactor? | Extract `EvidenceExtractor` base class with task-specific keyword registries |

### 5.2 Section routing

Implemented inside `evidence_extraction` via `SECTION_MARKERS` matching `berichte_mapper` stitch headings. Priority: Diagnosen > Jetziges Leiden > Epikrise > Prozedere.

**Hemorrhage:** OP reports may need `[OP-Bericht]`, `[Befund]`, admission/discharge sections — **remap sections**.

### 5.3 Prefiltering

If `not llm_should_receive_evidence(snippets)` → skip LLM unless short-report fallback.

**Delirium assumption:** No keywords ⇒ likely no delirium (`klasse=0`). **Dangerous for hemorrhage** if keywords are incomplete (silent negatives).

### 5.4 Guardrails (`clinical_guardrails.py`)

**Purpose:** Deterministic overrides after Agent 2 — hard-exclude prophylaxis-only, negation-only, isolated weak indirect symptoms; allow clusters and direct delir.

| Question | Answer |
|----------|--------|
| Reusable? | **No** |
| Refactor? | New `apply_*_guardrails` for hemorrhage with hemorrhage-specific rules (e.g. old healed bleed vs acute) |

### 5.5 Prompt construction

Loaded from `prompts/*.txt` at runtime (cwd = project root). German clinical language; **English JSON keys** for parser stability.

### 5.6 Inference orchestration (`run_pipeline.py`)

Sequential per-report processing (no batching, no parallel LLM). Writes CSV at end; optional SQLite row log.

### 5.7 Aggregation

- **Patient-level:** `max()` over report predictions (in analysis modules)
- **Patient × report-type matrix:** `patient_reporttype_matrix.py` for exploration

### 5.8 Evaluation

- **Baseline-centric:** `evaluate_predictions.py` vs ICD/ICDSC (exploratory for thesis)
- **Manual-centric:** `manual_validation_eval.py` (primary)

---

## 6. LLM system

### 6.1 Providers (`model_config.py`)

| Provider | Env | Default endpoint |
|----------|-----|------------------|
| `usz_api` | `LLM_PROVIDER=usz_api` | `USZ_LLM_URL=http://localhost:8100/generate` |
| `ollama` | `LLM_PROVIDER=ollama` | `OLLAMA_URL` chat API |

**Generation params:** `LLM_TEMPERATURE` (default 0.1; thesis uses 0), `LLM_TOP_P`, `LLM_MAX_TOKENS` (1000), `LLM_TIMEOUT` (120s)

### 6.2 Prompt pipeline

```text
Agent 1: system prompt (agent_extraction.txt) + user content = llm_report_text
Agent 2: system prompt (agent_interpretation.txt) + user = evidence + Agent1 JSON
Agent 3: Pure Python (classification.py) — no LLM
Guardrails: Pure Python (clinical_guardrails.py)
```

### 6.3 JSON schemas

**Agent 1 output:**
```json
{
  "desorientierung": [],
  "delir_explizit": [],
  "hyperaktivitaet_agitation": [],
  "vigilanz": [],
  "delir_therapie": [],
  "delir_prophylaxe": []
}
```

**Agent 2 output:**
```json
{
  "signalstaerke": "niedrig|mittel|hoch",
  "kontext": "",
  "alternative_erklaerung": false,
  "alternative_erklaerung_keywords": [],
  "begruendung": []
}
```

### 6.4 Parsing & error handling

1. `call_llm` → raw string
2. `extract_first_json_object` — brace matching, strips Gemma turn tokens
3. `parse_llm_json_output` — `json.loads`, regex `{.*}` fallback
4. On failure: `write_llm_debug()` → **empty safe defaults** (no retry)

**Agent failure defaults:** empty signal lists, `signalstaerke=niedrig` → usually `klasse=0`

### 6.5 Generic vs delirium-specific

| Component | Generic? |
|-----------|----------|
| `llm_interface.py`, `json_parsing.py`, `llm_debug.py` | **Yes** |
| Prompts, JSON field names, guardrails | **No** |
| Evidence bundle format | **No** |

---

## 7. Configuration system

### 7.1 File-based (`paths.py`)

| Constant | Current | Notes |
|----------|---------|-------|
| `DATA_MODE` | `"real"` | `"synthetic"` for offline tests |
| `BASELINE_COMPOSITE_MODE` | `"AND"` | **TEMPORARY** — thesis default documented as `"OR"` |
| `DEFAULT_MAX_REPORTS` | `None` | Full corpus |
| All `*_PATH` constants | — | Must remain single source of truth |

### 7.2 Environment variables (complete)

| Variable | Default | Module |
|----------|---------|--------|
| `MAX_REPORTS` | unset=all | `paths.py` |
| `VALIDATION_COHORT_ONLY` | false | `validation_cohort_filter.py` |
| `PATIENT_VALIDATION_N` | 100 | `export_patient_validation_cohort.py` |
| `OVERWRITE_FROZEN_VALIDATION` | false | `frozen_validation_cohort.py` |
| `LLM_PROVIDER`, `USZ_LLM_URL`, `OLLAMA_*`, `LLM_*` | see model_config | `model_config.py` |
| `DEBUG_LLM_OUTPUT` | false | pipeline, parsing |
| `ENABLE_SQLITE_LOGGING` | false | `run_pipeline.py` |
| `EVIDENCE_MAX_SNIPPETS` | 12 | `evidence_extraction.py` |
| `EVIDENCE_MAX_LLM_CHARS` | 8000 | `evidence_extraction.py` |
| `EVIDENCE_WINDOW_SENTENCES` | 1 | `evidence_extraction.py` |
| `EVIDENCE_MAX_SNIPPET_CHARS` | 400 | `evidence_extraction.py` |
| `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM` | false | `evidence_extraction.py` |
| `SHORT_REPORT_CHAR_THRESHOLD` | 1000 | `evidence_extraction.py` |

### 7.3 Hardcoded in source (not env)

| Location | Assumption |
|----------|------------|
| `run_pipeline.py` | `INPUT_MODE="berichte"`, `INTERPRETATION_MODE="prompt"` |
| `berichte_filters.py` | `REPORT_TYPES_FOR_MATRIX`, Dokumentationsblatt exclusion |
| `schema_normalize.py` | F05.0/F05.8/F05.9 allowlist; F05.1 excluded |
| `classification.py` | `mittel`/`hoch` → positive |
| `evidence_extraction.py` | All keyword tuples |
| `prompts/*.txt` | German delirium clinical definitions |

### 7.4 What should become generic vs task-specific

| Generic (shared package) | Task-specific (hemorrhage config) |
|--------------------------|-----------------------------------|
| Path layout pattern | Raw input filenames & columns |
| LLM transport & JSON parse | Prompts & schemas |
| Report identity & merge | Keyword/evidence registries |
| Validation cohort freeze/eval | Guardrails & classification rules |
| `tabular_io`, sqlite logging | Baseline construction |
| Env cap pattern (`EVIDENCE_*`) | Baseline composite definition |

---

## 8. Validation & evaluation

### 8.1 Evolution: baseline-centric → manual validation

**Early approach:** Compare model to ICD-10/ICDSC composite baseline; optimize agreement.

**Methodological failures discovered:**
1. **Baseline ≠ clinical truth** — ICDSC screening and ICD coding miss undocumented delirium; composite OR inflates "false positives" that are clinically plausible.
2. **Report-level preds vs patient-level baseline** — unfair FN/FP semantics.
3. **Excluding prefilter-skipped reports** — inflated performance (skipped rows are model decisions).
4. **`MAX_REPORTS` pilot runs** — broke patient completeness in validation cohort.
5. **Cohort regeneration after annotation** — invalidated manual work.

**Fixes implemented:**
- Frozen 100-patient cohort with **all** evaluatable reports per patient
- Report-level manual GT; patient GT **derived** via max()
- ICD/ICDSC as **reference signals only** in evaluation output
- `source_report_row_id` traceability
- `assert_spine_row_count_preserved` guards
- `evaluate_manual_validation` auto-prefers frozen files

### 8.2 Current validation strategy (thesis)

```text
1. Full run_pipeline (no MAX_REPORTS)
2. export_patient_validation_cohort (PATIENT_VALIDATION_N=100)
3. export_manual_report_labels
4. freeze_validation_cohort
5. Annotate manual_report_labels_frozen.csv (manual_report_ground_truth 0/1)
6. evaluate_manual_validation
```

**Primary metric:** Patient-level F1/sensitivity/specificity — `model_patient_positive` vs `derived_manual_patient_ground_truth`

**Secondary:** Report-level metrics for error analysis

### 8.3 Baseline evaluation (exploratory)

`evaluate_predictions.py` produces per-baseline binary confusion matrices under `outputs/evaluation/binary_baselines/`.

`BASELINE_COMPOSITE_MODE=AND` currently in `paths.py` — stricter than thesis OR mode.

### 8.4 Metrics (`manual_validation_eval.binary_metrics`)

Precision, recall, F1, sensitivity, specificity, accuracy from TP/TN/FP/FN on binary 0/1 series.

---

## 9. Current technical debt

### 9.1 Fragile components

- **No LLM retry** — single failure → negative prediction
- **No input chunking** — only char caps on evidence bundle; long inputs warn only
- **Sequential inference** — no parallelism; slow on full corpus
- **`pipeline_bericht` index-based** — fragile across CSV reorders
- **Legacy merge fallbacks** — mask ID problems instead of failing loud

### 9.2 Coupling

- Delirium keywords ↔ evidence ↔ prompts ↔ guardrails ↔ CSV column names (`delir_*`) — **tightly coupled**
- Analysis scripts import delirium-specific column names
- `compare_reports_vs_baseline` requires delirium baseline columns

### 9.3 Delirium hardcoding (dangerous for hemorrhage)

- Module names: `delirium_probability`, `delirium_hint_keywords`, `extract_delirium_evidence`
- All prompts and Agent JSON schemas
- ICD F05 + ICDSC entire baseline path
- `clinical_guardrails` symptom cluster logic
- Field analysis directories: `field_delirium/`

### 9.4 Problematic assumptions

- **Prefilter skip = negative** — may be wrong for hemorrhage with sparse keywords
- **Patient max aggregation** — any positive report → patient positive (may suit hemorrhage OR may need different rule e.g. admission-only)
- **German delirium-specific negation/prophylaxis handling** — not transferable
- **`BASELINE_COMPOSITE_MODE=AND`** left in code as "temporary" — easy to forget revert

### 9.5 Scalability / ops

- Full prediction CSV rewrite each run (no incremental)
- SQLite logging optional but not used in thesis path
- Docker path exists but secondary to local USZ API

### 9.6 Duplicated / legacy

- `report_text_llm_reduction.py` vs `evidence_extraction.py`
- `diagnosis_mapper.py` / `INPUT_MODE=diagnosis` — synthetic only
- `export_manual_validation_sample.py`, `run_error_review_export.py` — legacy 0/1/2 labels
- Multiclass `baseline_reference_class` still written

### 9.7 hemorrhage_project-specific debt

- Repo is a **verbatim fork** — name implies hemorrhage but code is 100% delirium
- Risk of running delirium experiments under wrong project folder
- No hemorrhage raw data contract documented yet

---

## 10. Reusability analysis

| Component | Generic | Delirium-specific | Reusable for hemorrhage | Refactor needed | Notes |
|-----------|---------|-------------------|-------------------------|-----------------|-------|
| `paths.py` layout | Partial | Paths names | Yes | Rename constants | Keep single source of truth |
| `tabular_io.py` | Yes | No | Yes | No | |
| `berichte_mapper.py` | Partial | Section map | Yes | Section headings for OP/Eintritt/Austritt |
| `report_identity.py` | Yes | No | Yes | No | Critical for validation |
| `berichte_filters.py` | Partial | Report type sets | Yes | Define hemorrhage report types |
| `evidence_extraction.py` | Pattern | Yes | No | **Rewrite** | New keyword types |
| `run_pipeline.py` | Pattern | Yes | Partial | Parameterize task hooks |
| `extraction.py` / `interpretation_llm.py` | Pattern | Yes | Partial | New prompts/schemas |
| `classification.py` | Pattern | Yes | Yes | Threshold labels |
| `clinical_guardrails.py` | No | Yes | No | **Rewrite** |
| `delirium_probability.py` | No | Yes | No | Remove or replace |
| `llm_interface.py` | Yes | No | Yes | No | |
| `json_parsing.py` | Yes | No | Yes | No | |
| `model_config.py` | Yes | No | Yes | No | |
| `prepare_structured_data.py` | No | Yes | No | **Replace** | Hemorrhage reference labels |
| `schema_normalize.py` | Partial | F05 rules | Partial | New code allowlist |
| `baseline_composite.py` | No | Yes | No | Replace | |
| `compare_reports_vs_baseline.py` | Pattern | Yes | Partial | New baseline columns |
| `evaluate_predictions.py` | Yes | Column names | Yes | Config-driven baseline cols |
| `validation_cohort_reports.py` | Yes | No | Yes | No | |
| `export_patient_validation_cohort.py` | Yes | Column names | Yes | Generalize delir_* cols |
| `freeze_validation_cohort.py` | Yes | No | Yes | No | |
| `manual_validation_eval.py` | Yes | No | Yes | No | max() aggregation pattern |
| `patient_reporttype_matrix.py` | Partial | bertyp | Partial | May add OP vs Eintritt |
| `validate_inputs.py` | Partial | Checks | Partial | New validation rules |
| `prompts/*.txt` | No | Yes | No | **Rewrite** | Hemorrhage clinical defs |
| `tests/` | Partial | Fixtures | Yes | Update fixtures |
| Analysis `run_field_delirium_*` | No | Yes | No | Remove/replace |
| `scripts/run_all.sh` | Pattern | Stages | Yes | Drop/replace baseline steps |

---

## 11. Execution workflow

### 11.1 Standard full run

```bash
cd hemorrhage_project   # after adaptation
./scripts/run_all.sh
```

**Stages:** `prepare_structured_data` → `run_pipeline` → `compare_reports_vs_baseline` → `evaluate_predictions` → `validate_inputs` → `run_exploration` → `run_analysis`

### 11.2 Thesis / manual validation run

```bash
unset MAX_REPORTS
export LLM_PROVIDER=usz_api
export LLM_TEMPERATURE=0
export SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true   # delirium-specific; revisit for hemorrhage

python3 -m src.pipeline.prepare_structured_data
python3 -m src.pipeline.run_pipeline
python3 -m src.analysis.run_validation_suite

export PATIENT_VALIDATION_N=100
python3 -m src.analysis.export_patient_validation_cohort
python3 -m src.analysis.export_manual_report_labels
python3 -m src.analysis.freeze_validation_cohort
# → annotate frozen labels
python3 -m src.analysis.evaluate_manual_validation
```

### 11.3 Cohort-only re-inference

```bash
export VALIDATION_COHORT_ONLY=true
python3 -m src.pipeline.run_pipeline
# → validation_cohort_predictions.csv (does NOT overwrite full predictions)
```

### 11.4 Preflight

```bash
./scripts/preflight_check.sh
```

---

## 12. Exports & outputs

| Directory | Contents |
|-----------|----------|
| `outputs/baseline/` | `structured_baseline.csv` |
| `outputs/predictions/` | `agent1_agent2_agent3_results_prompt.csv`, `agent_results_{provider}_{model}.csv`, optional `validation_cohort_predictions.csv` |
| `outputs/comparisons/` | `report_vs_baseline_comparison.csv`, excluded rows |
| `outputs/evaluation/` | Multiclass + `binary_baselines/` tables/plots |
| `outputs/validation/` | `validation_results.csv`, `validation_summary.txt` |
| `outputs/analysis/exploration/` | EDA tables/plots |
| `outputs/analysis/manual_validation/` | Cohort, labels, frozen/, evaluation/ |
| `outputs/analysis/patient_level/` | `patient_reporttype_matrix.csv` |
| `outputs/logs/llm_debug/` | Failed LLM payloads |

---

## 13. Methodological assumptions (delirium-embedded)

### 13.1 Must NOT transfer blindly to hemorrhage

| Assumption | Where embedded | Why dangerous for hemorrhage |
|------------|----------------|------------------------------|
| No keyword hit ⇒ skip LLM ⇒ negative | `run_pipeline`, `evidence_extraction` | Miss subclinical or atypical bleeding descriptions |
| Delirium keyword lexicon complete | `evidence_extraction.py` | Wrong disease vocabulary |
| Prophylaxis/screening ≠ disease | guardrails, prompts | Hemorrhage "risk" language may differ |
| Negation patterns for delirium | evidence + guardrails | Different clinical negation for bleed |
| `signalstaerke` ternary → binary | classification + guardrails | May need different confidence model |
| ICD F05 + ICDSC composite baseline | entire baseline path | Wrong reference standard |
| Patient positive = max(report preds) | `manual_validation_eval` | Hemorrhage may need case-level rule across OP+Eintritt+Austritt |
| Report types: Verlauf/Verlegung/Austritt only | `berichte_filters` | Hemorrhage needs **OP + Eintritts + Austritts** |
| Section stitch: diag/epikrise/jetziges_leiden/prozedere | `berichte_mapper` | OP reports need different fields |
| Exclude Dokumentationsblatt | `berichte_filters` | Re-evaluate for hemorrhage chart |
| Short-report fulltext fallback bertypen | `evidence_extraction` | OP reports may be long but keyword-sparse |
| German delirium therapy terms (Haldol etc.) | prompts, evidence | Replace with bleeding/imaging terms |
| F05.1 (alcohol withdrawal) excluded | `schema_normalize` | N/A |
| Manual GT = "delirium documented in this report" | annotation docs | Redefine for hemorrhagic vs non-hemorrhagic **case** |

### 13.2 May transfer with explicit redefinition

| Assumption | Condition |
|------------|-----------|
| Report-level annotation | If hemorrhage labels are per-report |
| Patient max aggregation | If "any report documents hemorrhage" defines case positive |
| Frozen cohort methodology | Strongly recommended |
| `source_report_row_id` | Strongly recommended |
| Evidence bundle → LLM | If keywords adapted; consider disabling prefilter initially |
| Binary `klasse` | If task is hemorrhagic vs non-hemorrhagic |

---

## 14. Hemorrhage transfer analysis

### 14.1 Target task (from project brief)

- Classify **hemorrhagic vs non-hemorrhagic** clinical cases
- Inputs: **OP reports**, **Eintrittsberichte**, **Austrittsberichte**
- Structured evidence extraction + local LLM
- **Case-level** classification with **manually curated reference labels**

### 14.2 Recommended architecture for hemorrhage

```text
hemorrhage_project/
  src/
    core/                    # NEW: shared from delirium (copy once)
      paths.py               # task-agnostic paths
      tabular_io.py
      report_identity.py
      llm_interface.py
      json_parsing.py
      validation_cohort/     # freeze, merge, eval (from analysis/)
    tasks/
      hemorrhage/
        config.py            # keywords, report types, sections, prompts path
        evidence_extraction.py
        agents/
        guardrails.py
        baseline.py          # manual labels CSV join, NOT ICDSC
        run_pipeline.py
    preprocessing/
      berichte_mapper.py     # parameterized section map
  prompts/hemorrhage/
    agent_extraction.txt
    agent_interpretation.txt
  data/raw/
    Berichte.csv
    reference_labels.csv     # patient or case level manual curated
```

### 14.3 Transfer strategy (phased)

#### Phase 0 — Repo hygiene (immediate)
- [ ] Add this document; mark fork status in README
- [ ] Stop running delirium prompts under `hemorrhage_project` name
- [ ] Create `TASK=hemorrhage` or separate entrypoint `run_hemorrhage_pipeline.py`

#### Phase 1 — Copy unchanged
- `report_identity.py`, `validation_cohort_reports.py`, `freeze_validation_cohort.py`, `manual_validation_eval.py`, `validation_ids.py`, `tabular_io.py`, `llm_interface.py`, `json_parsing.py`, `model_config.py`, sqlite logging, test patterns for merge integrity

#### Phase 2 — Generalize
- `paths.py` — remove ICDSC paths or make optional; add `REFERENCE_LABELS_PATH`
- `berichte_mapper` — configurable `SECTION_FIELDS` per bertyp
- `berichte_filters` — `REPORT_TYPES_HEMORRHAGE = (OP-Bericht, Eintrittsbericht, Austrittsbericht, …)`
- `export_patient_validation_cohort` — rename `delir_*` columns to neutral `model_*` / `task_*`
- `run_pipeline` — inject task-specific `extract_evidence`, `guardrails`, `classify`

#### Phase 3 — Rewrite (delirium-specific)
- `evidence_extraction.py` → hemorrhage keyword groups: `direct_bleed`, `imaging_hemorrhage`, `postop_bleeding`, `negation`, `historical_bleed`
- `prompts/*` → hemorrhage clinical definitions (German)
- `clinical_guardrails.py` → acute vs chronic, postoperative vs traumatic, excluded incidental findings
- `prepare_structured_data.py` → join **manual reference labels** only (no F05/ICDSC unless exploratory)
- Remove or quarantine: `delirium_probability`, `run_field_delirium_analysis`, ICDSC overlap scripts

#### Phase 4 — Case-level classification
- Define **case** entity (likely `PatientenID` or study-specific `case_id`)
- Explicit aggregation policy, e.g.:
  - `case_positive = max(report_positive)` OR
  - `case_positive = any(OP positive) OR any(Eintritt positive)` — **must be decided clinically**
- Manual labels at case level in `reference_labels.csv`; derive report-level evaluation separately

#### Phase 5 — Disable risky defaults initially
- Set `SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true` OR disable prefilter skip until keyword coverage validated
- Log all `status=skipped` for manual review

### 14.4 What to remove from hemorrhage repo (eventually)

- ICDSC-specific analysis scripts
- Delirium field analysis (`field_delirium/`)
- Legacy `diagnosis_mapper` path (unless needed for tests)
- `delirium_hint_keywords.py` after hemorrhage replacement

### 14.5 What NOT to do

- Do not copy `BASELINE_COMPOSITE_MODE` / ICDSC logic as "ground truth"
- Do not rename files only (cosmetic) without changing prompts and evidence
- Do not skip frozen cohort for manual validation
- Do not use `MAX_REPORTS` for validation runs

---

## 15. Current project state & risks

### 15.1 hemorrhage_project state (2026-05-23)

| Item | Status |
|------|--------|
| Codebase | Near-identical fork of `delirium_project` |
| Delirium-specific logic | **100% present** — prompts, evidence, guardrails, baselines |
| Hemorrhage implementation | **Not started** |
| `HANDOVER_SUMMARY.md` | Operational delirium runbook (copied) |
| Tests | Delirium fixtures — will fail semantically until adapted |
| Raw data | Not in git (`data/` gitignored) |

### 15.2 Known risks for transfer

1. **False confidence** — running existing pipeline produces "results" that are delirium, not hemorrhage
2. **Prefilter silent negatives** — highest risk if keywords not rebuilt
3. **Wrong report types in cohort** — validation export filters wrong bertyp set
4. **Patient max aggregation** — may not match case-level hemorrhage definition
5. **Baseline confusion** — ICD/ICDSC columns in cohort export may be mistaken for GT
6. **Documentation drift** — `HANDOVER_SUMMARY.md` vs this file; prefer **this file** for engineering decisions

### 15.3 Recommended next steps (implementation — out of scope for this doc)

1. Define hemorrhage raw CSV schema (columns for OP/Eintritt/Austritt)
2. Define manual `reference_labels.csv` schema (case vs report level)
3. Implement `tasks/hemorrhage/evidence_extraction.py` with pilot keyword list
4. Draft prompts with hemorrhage JSON schema
5. Disable or relax prefilter; run small `MAX_REPORTS=50` pilot
6. Port validation cohort machinery with new column names
7. Add tests for hemorrhage evidence types (copy pattern from `tests/test_evidence_extraction.py`)

### 15.4 Related documents

| File | Role |
|------|------|
| `HANDOVER_SUMMARY.md` | Short operational runbook (delirium commands) |
| `README.md` / `RUNBOOK.md` | User-facing setup |
| `PROJECT_STATUS.md` | Minimal snapshot |
| `delirium_project/TECHNICAL_HANDOVER.md` | **Optional:** copy/sync if maintaining both repos |

---

## Appendix A: Agent 3 + guardrail decision flow (delirium)

```text
Agent 2 signalstaerke
        │
        ▼
classify_delirium: mittel|hoch → preliminary klasse=1
        │
        ▼
clinical_guardrails (overrides):
  - no_evidence_prefilter_skip → klasse=0
  - prophylaxis_only → klasse=0
  - negated_delir (no explicit positive) → klasse=0
  - direct_delir → klasse=1
  - delir_therapy + context → klasse=1
  - isolated_indirect → klasse=0 (manual_review)
  - alternative_explanation without cluster → klasse=0
  - symptom_cluster → klasse=1 (often manual_review)
        │
        ▼
Final klasse + decision_rule_applied
```

---

## Appendix B: Environment quick reference (thesis delirium run)

```bash
unset MAX_REPORTS
export LLM_PROVIDER=usz_api
export LLM_TEMPERATURE=0
export LLM_TOP_P=1
export SEND_SHORT_REPORTS_WITHOUT_EVIDENCE_TO_LLM=true
export SHORT_REPORT_CHAR_THRESHOLD=1000
export DEBUG_LLM_OUTPUT=false
export ENABLE_SQLITE_LOGGING=true
```

---

*End of technical handover.*
