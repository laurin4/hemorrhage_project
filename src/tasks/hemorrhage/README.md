# Hemorrhage task — case-centric pipeline

## Phase 0b — Real data inspection (current)

Structural validation of server Excel files — **no NLP**.

```bash
python3 -m src.tasks.hemorrhage.inspect_data
```

**Inputs** (defaults under `data/raw/`):

| Role | Default filename |
|------|------------------|
| Clinical reports | `260507_CCM_DAVF.xlsx` |
| Reference labels | `NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx` |

Env: `HEMORRHAGE_REPORTS_XLSX`, `HEMORRHAGE_REFERENCE_XLSX`, optional sheet names.

**Outputs:** `data/inspection/` (CSVs + `inspection_summary.txt`)

## Phase 0 — Case build from CSV

```bash
python3 -m src.tasks.hemorrhage.build_cases --input data/raw/reports.csv
```

### Required input columns

- `excel_pid`, `excel_opdat`, `opber_fallnr` — case keys
- `typus` — e.g. `01 Operationsbericht`, `02 Eintrittsbericht`, `03 Austrittsbericht`
- Text: `diag`, `indik_untersuch`, `vorgehen_beurt` (CCM export)

### Prefilter

Default: `HEMORRHAGE_PREFILTER_MODE=disabled` — do not use delirium keyword skip for hemorrhage.
