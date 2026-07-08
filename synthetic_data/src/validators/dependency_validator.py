"""
validators/dependency_validator.py - Logical dependency consistency checks.

Verifies that relationships between features are logically consistent.
For example, it checks if high-income households rarely have poor sanitation.
"""

import pandas as pd
from typing import Any

from synthetic_data.src.validators.base import BaseValidator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyValidator(BaseValidator):
    """
    Checks if proportions of anomalous logical combinations exceed configured limits.
    """

    def validate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        dependency_rules = self.val_config.get("dependency_rules", [])

        for rule in dependency_rules:
            name = rule.get("name")
            cond = rule.get("condition", {})
            max_prop = rule.get("max_proportion", 0.05)
            severity = rule.get("severity", "warning")

            # Check if all columns in condition exist in DataFrame
            if not all(col in df.columns for col in cond):
                continue

            # Build query filter for condition
            # Form: (df[col1] == val1) & (df[col2] == val2)
            filters = []
            for col, val in cond.items():
                filters.append(df[col] == val)

            if not filters:
                continue

            # Total matching anomalous condition
            anomaly_mask = filters[0]
            for f in filters[1:]:
                anomaly_mask = anomaly_mask & f

            anomaly_count = anomaly_mask.sum()

            # We calculate proportion relative to the base condition
            # For example, "proportion of high income households that have poor sanitation"
            # The base is the first condition key-value pair, or the whole dataset.
            # Let's calculate relative to the first condition column to be specific.
            base_col = list(cond.keys())[0]
            base_val = cond[base_col]
            base_count = (df[base_col] == base_val).sum()

            if base_count == 0:
                continue

            prop = anomaly_count / base_count

            if prop > max_prop:
                issues.append(
                    {
                        "rule": f"dependency_consistency:{name}",
                        "message": (
                            f"Anomalous combination {cond} appears in {anomaly_count} records "
                            f"({prop:.2%} of '{base_col}={base_val}'), exceeding limit of {max_prop:.2%}."
                        ),
                        "severity": severity,
                        "meta": {
                            "anomaly_count": int(anomaly_count),
                            "base_count": int(base_count),
                            "proportion": float(prop),
                            "limit": max_prop,
                        },
                    }
                )

        return issues
