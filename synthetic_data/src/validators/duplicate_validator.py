"""
validators/duplicate_validator.py - Duplicate row checks.

Verifies that the proportion of identical rows does not exceed thresholds.
"""

import pandas as pd
from typing import Any

from synthetic_data.src.validators.base import BaseValidator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateValidator(BaseValidator):
    """
    Checks for identical rows in the generated dataset.
    """

    def validate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        dup_cfg = self.val_config.get("duplicate_rules", {})
        check_dup = dup_cfg.get("check_exact_duplicates", True)
        max_ratio = dup_cfg.get("max_duplicate_ratio", 0.01)
        error_on_exceed = dup_cfg.get("error_on_exceed", False)

        if not check_dup:
            return issues

        # Count exact duplicates (excluding the first occurrence)
        dup_count = df.duplicated().sum()
        ratio = dup_count / len(df)

        if ratio > max_ratio:
            severity = "error" if error_on_exceed else "warning"
            issues.append(
                {
                    "rule": "duplicate_rows",
                    "message": (
                        f"Dataset has {dup_count} duplicate rows ({ratio:.2%}), "
                        f"exceeding max duplicate ratio of {max_ratio:.2%}."
                    ),
                    "severity": severity,
                    "meta": {"duplicate_count": int(dup_count), "ratio": float(ratio)},
                }
            )

        return issues
