"""
validators/distribution_validator.py - Distribution balance checks.

Compares the target class distribution (risk_level) against configured expected values.
"""

import pandas as pd
from typing import Any

from synthetic_data.src.validators.base import BaseValidator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class DistributionValidator(BaseValidator):
    """
    Checks if target or feature distributions deviate significantly from expectations.
    """

    def validate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        dist_rules = self.val_config.get("distribution_rules", {})

        for col, rule in dist_rules.items():
            if col not in df.columns:
                continue

            expected = rule.get("expected", {})
            tolerance = rule.get("tolerance", 0.10)
            severity = rule.get("severity", "warning")

            total_non_null = df[col].notnull().sum()
            if total_non_null == 0:
                continue

            actual_probs = df[col].value_counts(normalize=True).to_dict()

            for cat, exp_prob in expected.items():
                act_prob = actual_probs.get(cat, 0.0)
                diff = abs(act_prob - exp_prob)

                if diff > tolerance:
                    issues.append(
                        {
                            "rule": f"distribution_drift:{col}:{cat}",
                            "message": (
                                f"Distribution of '{col}' for category '{cat}' is {act_prob:.2%}, "
                                f"deviating from expected {exp_prob:.2%} by {diff:.2%} (tolerance: {tolerance:.2%})."
                            ),
                            "severity": severity,
                            "meta": {
                                "category": cat,
                                "expected": float(exp_prob),
                                "actual": float(act_prob),
                                "deviation": float(diff),
                            },
                        }
                    )

        return issues
