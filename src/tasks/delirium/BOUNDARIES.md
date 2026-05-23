# Delirium task boundaries (legacy)

**Do not use these modules for hemorrhage case-centric inference.**

## Delirium-specific (isolated — not deleted in Phase 0)

| Area | Location | Notes |
|------|----------|-------|
| Evidence / keywords | `src/preprocessing/evidence_extraction.py`, `delirium_hint_keywords.py` | Keyword prefilter, delir lexicon |
| Agents | `src/agents/extraction.py`, `interpretation_llm.py`, `classification.py`, `clinical_guardrails.py` | Prompts + guardrails |
| Probability | `src/agents/delirium_probability.py` | Exploratory score |
| Baseline | `src/pipeline/prepare_structured_data.py`, `schema_normalize.py` (F05), `baseline_composite.py` | ICD-10 F05 + ICDSC |
| Prompts | `prompts/agent_*.txt` | German delirium instructions |
| Analysis | `src/analysis/run_field_delirium_analysis.py`, ICDSC overlap scripts | Delirium EDA |
| CSV columns | `delir_*`, `delirium_*` in predictions export | Report-level |

## Report-centric (see `src/core/legacy/report_centric.py`)

- `run_pipeline.py` — one report → one prediction
- `export_patient_validation_cohort.py` — report-level validation rows
- `manual_validation_eval.compute_model_patient_positive` — max over reports

## Running delirium pipeline unchanged

```bash
export PROJECT_TASK=delirium
python3 -m src.pipeline.run_pipeline
```

Hemorrhage case construction does **not** invoke the above.
