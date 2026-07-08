"""
validators/range_validator.py - Range, category, binary, and missing value checks.

Verifies that:
    - Numeric values fall within configured [min, max] limits.
    - Categorical values belong to the allowed lists.
    - Binary values are strictly 0, 1, or None.
    - Missing value ratios do not exceed thresholds, and required columns have no missing values.
"""

import pandas as pd
from typing import Any

from synthetic_data.src.validators.base import BaseValidator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class RangeValidator(BaseValidator):
    """
    Validates limits on ranges, categorical sets, binaries, and missing values.
    """

    def validate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []

        # 1. Range Validation
        range_rules = self.val_config.get("range_rules", {})
        for col, rule in range_rules.items():
            if col not in df.columns:
                continue

            series = df[col].dropna()
            c_min, c_max = rule["min"], rule["max"]

            # Check bounds
            violations_min = series[series < c_min]
            violations_max = series[series > c_max]

            if len(violations_min) > 0 or len(violations_max) > 0:
                issues.append(
                    {
                        "rule": f"range_limits:{col}",
                        "message": (
                            f"Column '{col}' has {len(violations_min) + len(violations_max)} "
                            f"violations of limits [{c_min}, {c_max}]."
                        ),
                        "severity": "error",
                        "meta": {
                            "min_limit": c_min,
                            "max_limit": c_max,
                            "actual_min": series.min(),
                            "actual_max": series.max(),
                        },
                    }
                )

        # 2. Category Validation
        category_rules = self.val_config.get("category_rules", {})
        for col, rule in category_rules.items():
            if col not in df.columns:
                continue

            series = df[col].dropna()
            allowed = set(rule["allowed"])
            invalid = series[~series.isin(allowed)]

            if len(invalid) > 0:
                issues.append(
                    {
                        "rule": f"category_limits:{col}",
                        "message": (
                            f"Column '{col}' has {len(invalid)} invalid categories: "
                            f"{invalid.unique().tolist()}. Allowed: {list(allowed)}."
                        ),
                        "severity": "error",
                        "meta": {"allowed": list(allowed), "invalid_found": invalid.unique().tolist()},
                    }
                )

        # 3. Binary Validation
        binary_rules = self.val_config.get("binary_rules", [])
        for col in binary_rules:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            invalid = series[~series.isin({0, 1, 0.0, 1.0})]

            if len(invalid) > 0:
                issues.append(
                    {
                        "rule": f"binary_limits:{col}",
                        "message": f"Column '{col}' has {len(invalid)} values that are not 0 or 1.",
                        "severity": "error",
                        "meta": {"invalid_count": len(invalid)},
                    }
                )

        # 4. Missing Value Validation
        missing_cfg = self.val_config.get("missing_value_rules", {})
        max_ratio = missing_cfg.get("max_missing_ratio_per_column", 0.05)
        no_missing_allowed = missing_cfg.get("columns_no_missing_allowed", [])

        # Check max ratio per column
        for col in df.columns:
            null_count = df[col].isnull().sum()
            ratio = null_count / len(df)

            if ratio > max_ratio:
                issues.append(
                    {
                        "rule": f"missing_ratio_excess:{col}",
                        "message": (
                            f"Column '{col}' has {null_count} missing values ({ratio:.2%}), "
                            f"exceeding max ratio of {max_ratio:.2%}."
                        ),
                        "severity": "warning",
                        "meta": {"null_count": int(null_count), "ratio": float(ratio)},
                    }
                )

        # Check required columns
        for col in no_missing_allowed:
            if col not in df.columns:
                continue

            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(
                    {
                        "rule": f"missing_not_allowed:{col}",
                        "message": f"Column '{col}' is required but has {null_count} missing values.",
                        "severity": "error",
                        "meta": {"null_count": int(null_count)},
                    }
                )

        return issues
