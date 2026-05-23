"""
Deterministic case identifiers from clinical grouping keys.

Case = (excel_pid, excel_opdat, opber_fallnr)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Tuple

MISSING_KEY_TOKEN = "__MISSING__"

_CASE_ID_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass(frozen=True)
class CaseKey:
    excel_pid: str
    excel_opdat: str
    opber_fallnr: str

    def parts(self) -> Tuple[str, str, str]:
        return (self.excel_pid, self.excel_opdat, self.opber_fallnr)

    def has_missing_component(self) -> bool:
        return MISSING_KEY_TOKEN in self.parts()

    @classmethod
    def from_row(cls, row: object, *, columns: Tuple[str, str, str]) -> "CaseKey":
        """Build a key from a pandas Series or dict-like row."""
        pid_col, opdat_col, fallnr_col = columns

        def _get(col: str) -> str:
            if hasattr(row, "get"):
                raw = row.get(col, "")
            else:
                raw = getattr(row, col, "")
            return normalize_case_key_part(raw)

        return cls(
            excel_pid=_get(pid_col),
            excel_opdat=_get(opdat_col),
            opber_fallnr=_get(fallnr_col),
        )


def normalize_case_key_part(value: object) -> str:
    """Strip and normalize one case key component; empty/NaN → ``MISSING_KEY_TOKEN``."""
    if value is None:
        return MISSING_KEY_TOKEN
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return MISSING_KEY_TOKEN
    return s


def _slug_part(part: str, max_len: int = 48) -> str:
    slug = _CASE_ID_SLUG_RE.sub("_", part).strip("_")
    if not slug:
        slug = "empty"
    return slug[:max_len]


def compute_case_id(key: CaseKey, *, style: str = "readable") -> str:
    """
  Return a stable ``case_id`` for *key*.

  - ``readable``: ``case_{pid}_{opdat}_{fallnr}`` (sanitized, truncated parts)
  - ``hash``: ``case_{sha256[:16]}`` (compact; use when parts are long or sensitive)
    """
    pid, opdat, fallnr = key.parts()
    if style == "hash":
        digest = hashlib.sha256("|".join((pid, opdat, fallnr)).encode("utf-8")).hexdigest()
        return f"case_{digest[:16]}"
    return "case_{}__{}__{}".format(
        _slug_part(pid),
        _slug_part(opdat),
        _slug_part(fallnr),
    )
