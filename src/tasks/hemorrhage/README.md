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
