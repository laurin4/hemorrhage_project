"""
Registry of modules that assume **one report = one prediction**.

Hemorrhage uses **one case = one prediction**. These modules remain for delirium
compatibility and must not be used for hemorrhage inference without refactoring.

DO NOT import this registry into production hemorrhage paths — documentation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ReportCentricAssumption:
    module: str
    assumption: str
    hemorrhage_status: str  # blocked | refactor_required | reference_only


REPORT_CENTRIC_MODULES: Tuple[ReportCentricAssumption, ...] = (
    ReportCentricAssumption(
        "src.pipeline.run_pipeline",
        "Main loop iterates report records; writes one prediction row per report.",
        "blocked",
    ),
    ReportCentricAssumption(
        "src.preprocessing.berichte_mapper.build_report_level_berichte_records",
        "One dict per Berichte row — report-level prediction unit.",
        "blocked",
    ),
    ReportCentricAssumption(
        "src.analysis.manual_validation_eval.compute_model_patient_positive",
        "Aggregates report predictions via max() per patient — delirium validation pattern.",
        "refactor_required",
    ),
    ReportCentricAssumption(
        "src.analysis.validation_cohort_reports.build_complete_validation_reports_frame",
        "Validation spine is one row per report; LEFT merge predictions per report.",
        "refactor_required",
    ),
    ReportCentricAssumption(
        "src.analysis.export_patient_validation_cohort",
        "Exports report-level rows with model_report_prediction.",
        "refactor_required",
    ),
    ReportCentricAssumption(
        "src.pipeline.compare_reports_vs_baseline",
        "Compares report-level predictions to patient-level baseline (duplicated per report).",
        "refactor_required",
    ),
    ReportCentricAssumption(
        "src.analysis.patient_reporttype_matrix",
        "Patient × bertyp matrix from report-level predictions.",
        "reference_only",
    ),
    ReportCentricAssumption(
        "src.preprocessing.evidence_extraction.extract_delirium_evidence",
        "Evidence + prefilter per single report text.",
        "blocked",
    ),
    ReportCentricAssumption(
        "src.preprocessing.report_identity.assign_source_report_row_id",
        "Report-row identity for merge — still useful inside case reports, not as prediction key.",
        "reference_only",
    ),
)


def list_blocked_for_hemorrhage() -> List[ReportCentricAssumption]:
    return [m for m in REPORT_CENTRIC_MODULES if m.hemorrhage_status == "blocked"]
