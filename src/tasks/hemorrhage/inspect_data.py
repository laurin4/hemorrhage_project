"""
CLI: load real hemorrhage Excel inputs, build cases, export inspection artifacts.

  python3 -m src.tasks.hemorrhage.inspect_data
  python3 -m src.tasks.hemorrhage.inspect_data --reports data/raw/NCH_pidlist_opdat_ab_eb_op_SJO_pg_DRQ0001416.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.paths import INSPECTION_DIR
from src.tasks.hemorrhage.inspection.runner import run_full_inspection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structural inspection of hemorrhage raw Excel data (no NLP)."
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=None,
        help="Clinical reports Excel (default: NCH_pidlist_...xlsx)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference / labels Excel (default: 260507_CCM_DAVF.xlsx)",
    )
    parser.add_argument("--output-dir", type=Path, default=INSPECTION_DIR, help="Inspection output directory")
    args = parser.parse_args(argv)

    result = run_full_inspection(
        reports_path=args.reports,
        reference_path=args.reference,
        output_dir=args.output_dir,
    )

    for line in result.summary_lines:
        print(line)

    if result.errors and result.cases_built == 0 and not result.reports_load:
        return 1
    if result.errors:
        LOGGER.warning("Inspection completed with %d error entries (see inspection_anomalies.txt)", len(result.errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
