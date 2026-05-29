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
| `--limit N` | First N cases |
| `--case-id ID` | Single case |
| `--output PATH` | Custom CSV path |
| `--reports` / `--reference` | Override Excel paths |

Prompt: `prompts/hemorrhage_case_classification.txt` (German, structured JSON).

### Two-level classification (supervisor clarification)

- **Level 1:** `hämorrhagisch` vs. `nicht_hämorrhagisch` (`klasse` 1 / 0).
- **Level 2 (only if hämorrhagisch):** `haemorrhage_subtype` ∈ {`akut`, `historisch`, `nicht_akut`}.
  - `nicht_hämorrhagisch` → `haemorrhage_subtype = null`.
  - hämorrhagisch but subtype missing/unclear → `haemorrhage_subtype = "unbekannt"` + uncertainty flag (no parse failure).
- **`Verify_Vaskulär` is metadata only**, never a class label. It must not influence the model decision and is excluded from binary TP/TN/FP/FN (unless `--include-verify-as-negative` sensitivity mode).
- **Binary evaluation is unchanged** (hemorrhagic vs non_hemorrhagic). **Subtype analysis is descriptive only** — no validated reference subtype labels exist yet, so subtype accuracy is not computed.

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
