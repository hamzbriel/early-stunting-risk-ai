"""
exporters/statistics_exporter.py - Exporting descriptive stats, risk summary, and correlations.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticsExporter:
    """
    Computes statistics, correlations, risk level distribution,
    and relationships matrix, then exports them as JSON artifacts.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.exp_config: dict[str, Any] = config.export.get("export", {})

    def export(self, df: pd.DataFrame) -> None:
        """
        Compute statistics, risk levels, and relationships matrices,
        then save them as JSON.
        """
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        files = self.exp_config.get("files", {})

        # 1. General Descriptive Statistics
        stats = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                stats[col] = {
                    "type": "numeric",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "q25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "missing_count": int(series.isnull().sum()),
                }
            else:
                counts = series.value_counts(dropna=False).to_dict()
                # Handle potential NaN keys
                processed_counts = {}
                for k, v in counts.items():
                    key_str = "NaN" if pd.isnull(k) else str(k)
                    processed_counts[key_str] = int(v)

                stats[col] = {
                    "type": "categorical",
                    "value_counts": processed_counts,
                    "unique_values": int(series.nunique()),
                    "missing_count": int(series.isnull().sum()),
                }

        stats_path = output_dir / files.get("statistics", "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        logger.info("Exported JSON statistics: %s", stats_path.resolve())

        # 2. Risk Summary
        risk_summary = {}
        if "risk_score" in df.columns and "risk_level" in df.columns:
            scores = df["risk_score"].dropna()
            levels = df["risk_level"].dropna()

            risk_summary = {
                "risk_score_stats": {
                    "mean": float(scores.mean()),
                    "std": float(scores.std()),
                    "min": float(scores.min()),
                    "median": float(scores.median()),
                    "max": float(scores.max()),
                },
                "risk_level_distribution": {
                    k: {
                        "count": int(v),
                        "percentage": float(v / len(df)),
                    }
                    for k, v in levels.value_counts().to_dict().items()
                },
            }

            # Average risk score by key categorical variables
            categorical_breakdowns = {}
            cols_to_check = ["gender", "family_income", "sanitation", "protein_intake", "immunization"]
            for col in cols_to_check:
                if col in df.columns:
                    breakdown = df.groupby(col, observed=True)["risk_score"].mean().to_dict()
                    categorical_breakdowns[col] = {str(k): float(v) for k, v in breakdown.items()}

            risk_summary["risk_score_by_features"] = categorical_breakdowns

        risk_path = output_dir / files.get("risk_summary", "risk_summary.json")
        with open(risk_path, "w", encoding="utf-8") as f:
            json.dump(risk_summary, f, indent=4)
        logger.info("Exported JSON risk summary: %s", risk_path.resolve())

        # 3. Relationships and Correlations
        # Build numeric correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().fillna(0).to_dict()

        relationships = {
            "pearson_correlations": corr_matrix,
            "configured_dag_edges": self.config.relationships.get("edges", []),
        }

        rel_path = output_dir / files.get("relationships", "relationships.json")
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(relationships, f, indent=4)
        logger.info("Exported JSON relationships manifest: %s", rel_path.resolve())
