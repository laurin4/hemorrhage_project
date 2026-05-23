"""
Case-first data structures for hemorrhage (and future case-centric tasks).

One ``ClinicalCase`` aggregates zero or more reports; missing report types are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.case.keys import CaseKey, compute_case_id


@dataclass
class CaseReport:
    """One report document attached to a case (may be one of several typus slots)."""

    typus_code: str
    typus_label: str
    report_text: str
    source_row_id: str = ""
    raw_typus: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClinicalCase:
    """
    Clinical case = (excel_pid, excel_opdat, opber_fallnr).

    ``reports`` is keyed by canonical typus code (``01``, ``02``, ``03``).
    Missing slots are listed in ``missing_report_types`` — not implied by absence alone
    during construction (constructor sets both available and missing explicitly).
    """

    case_key: CaseKey
    case_id: str
    reports: Dict[str, CaseReport] = field(default_factory=dict)
    available_report_types: Tuple[str, ...] = ()
    missing_report_types: Tuple[str, ...] = ()
    unexpected_report_types: Tuple[str, ...] = ()
    anomalies: List[str] = field(default_factory=list)
    extra_reports: Dict[str, CaseReport] = field(default_factory=dict)

    @property
    def excel_pid(self) -> str:
        return self.case_key.excel_pid

    @property
    def excel_opdat(self) -> str:
        return self.case_key.excel_opdat

    @property
    def opber_fallnr(self) -> str:
        return self.case_key.opber_fallnr

    @property
    def n_reports_available(self) -> int:
        return len(self.reports)

    @property
    def is_complete(self) -> bool:
        return len(self.missing_report_types) == 0 and self.n_reports_available > 0

    def get_report_text(self, typus_code: str) -> str:
        rep = self.reports.get(typus_code)
        return rep.report_text if rep else ""

    def structured_case_text(self) -> str:
        """Concatenate available reports with typus headings (no NLP)."""
        blocks: List[str] = []
        for code in sorted(self.reports.keys()):
            rep = self.reports[code]
            if rep.report_text.strip():
                blocks.append(f"[{rep.typus_label}]\n{rep.report_text.strip()}")
        return "\n\n".join(blocks)


@dataclass
class CaseConstructionStats:
    """Summary counters from flat-row → case grouping."""

    input_rows: int = 0
    cases_built: int = 0
    rows_without_text: int = 0
    rows_with_missing_key_component: int = 0
    duplicate_typus_in_case: int = 0
    unexpected_typus_rows: int = 0
    cases_with_zero_reports: int = 0
    cases_incomplete: int = 0
    cases_complete: int = 0
    anomaly_messages: List[str] = field(default_factory=list)

    def to_summary_lines(self) -> List[str]:
        return [
            "Case construction summary",
            f"  input_rows={self.input_rows}",
            f"  cases_built={self.cases_built}",
            f"  cases_complete={self.cases_complete}",
            f"  cases_incomplete={self.cases_incomplete}",
            f"  cases_with_zero_reports={self.cases_with_zero_reports}",
            f"  rows_without_text={self.rows_without_text}",
            f"  rows_with_missing_key_component={self.rows_with_missing_key_component}",
            f"  duplicate_typus_in_case={self.duplicate_typus_in_case}",
            f"  unexpected_typus_rows={self.unexpected_typus_rows}",
        ]


def build_clinical_case(
    key: CaseKey,
    reports: Dict[str, CaseReport],
    *,
    expected_typus_codes: Sequence[str],
    unexpected: Optional[Dict[str, CaseReport]] = None,
    anomalies: Optional[List[str]] = None,
    case_id_style: str = "readable",
) -> ClinicalCase:
    """Assemble a ``ClinicalCase`` with explicit available/missing typus lists."""
    expected = tuple(expected_typus_codes)
    available = tuple(sorted(reports.keys()))
    missing = tuple(code for code in expected if code not in reports)
    unexpected_keys = tuple(sorted((unexpected or {}).keys()))
    return ClinicalCase(
        case_key=key,
        case_id=compute_case_id(key, style=case_id_style),
        reports=reports,
        available_report_types=available,
        missing_report_types=missing,
        unexpected_report_types=unexpected_keys,
        anomalies=list(anomalies or []),
        extra_reports=dict(unexpected or {}),
    )
