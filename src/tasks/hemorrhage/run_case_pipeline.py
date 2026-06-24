"""
CLI: hemorrhage case-level LLM inference (prototype, qualitative review).

  python3 -m src.tasks.hemorrhage.run_case_pipeline --dry-run --limit 5
  python3 -m src.tasks.hemorrhage.run_case_pipeline --limit 5
  python3 -m src.tasks.hemorrhage.run_case_pipeline                 # ALL patients
  python3 -m src.tasks.hemorrhage.run_case_pipeline --labeled-only  # eval cohort only

By default every patient is classified (incl. Verify_Vaskulär / unlabeled). Use
--labeled-only to restrict to the binary-labeled cohort for evaluation runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.paths import HEMORRHAGE_CASE_PREDICTIONS_PATH
from src.tasks.hemorrhage.inference.runner import run_hemorrhage_case_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hemorrhage case-level LLM inference (one case = one prediction)."
    )
    parser.add_argument("--reports", type=Path, default=None, help="Reports Excel input")
    parser.add_argument("--reference", type=Path, default=None, help="Reference Excel (optional)")
    parser.add_argument("--output", type=Path, default=HEMORRHAGE_CASE_PREDICTIONS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N cases")
    parser.add_argument("--case-id", type=str, default=None, help="Process only one case_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts only; do not call LLM",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help=(
            "Restrict to the binary-labeled evaluation cohort "
            "(reference hemorrhagic / non_hemorrhagic only). "
            "Default: classify ALL patients, including Verify_Vaskulär / unlabeled."
        ),
    )
    parser.add_argument(
        "--include-verify-only",
        action="store_true",
        help="With --labeled-only, also include verify_only cases",
    )
    # Backwards-compatible alias (all-cases is now the default).
    parser.add_argument("--all-cases", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # Default behaviour: classify every patient (verify_only / unlabeled included).
    # --labeled-only opts back into the binary cohort for evaluation runs.
    process_all_cases = not args.labeled_only

    result = run_hemorrhage_case_pipeline(
        reports_path=args.reports,
        reference_path=args.reference,
        output_path=args.output,
        limit=args.limit,
        case_id=args.case_id,
        dry_run=args.dry_run,
        process_all_cases=process_all_cases,
        include_verify_only=args.include_verify_only,
    )

    for line in result.summary_lines:
        print(line)

    if result.cases_processed == 0:
        return 1
    if result.llm_failed_count == result.cases_processed and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
