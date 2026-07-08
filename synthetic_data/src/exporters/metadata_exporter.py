"""
exporters/metadata_exporter.py - Exporting dataset schema and build metadata to JSON.
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExporter:
    """
    Exports metadata schema, build stats, and validator logs.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.exp_config: dict[str, Any] = config.export.get("export", {})

    def export(self, datasets: dict[str, pd.DataFrame], validation_report: dict) -> None:
        """
        Save build metadata.
        """
        output_dir = self.config.output_dir
        file_name = self.exp_config.get("files", {}).get("metadata", "metadata.json")
        dest_path = output_dir / file_name

        total_samples = sum(len(df) for df in datasets.values())

        # Build schema definitions based on columns
        schema = {}
        sample_df = list(datasets.values())[0] if datasets else pd.DataFrame()
        for col in sample_df.columns:
            # Map type to simple descriptive string
            dtype = str(sample_df[col].dtype)
            if dtype.startswith("int"):
                val_type = "integer"
            elif dtype.startswith("float"):
                val_type = "float"
            elif dtype == "object":
                val_type = "categorical/string"
            else:
                val_type = dtype

            schema[col] = {
                "type": val_type,
                "null_count": int(sum(df[col].isnull().sum() for df in datasets.values())),
            }

        metadata = {
            "project_name": "early-stunting-risk-ai",
            "version": self.config.generator.version,
            "description": self.config.generator.description,
            "generated_at": datetime.now().isoformat(),
            "generator_seed": self.config.generator.seed,
            "total_records": total_samples,
            "split_info": {
                name: {
                    "count": len(df),
                    "ratio": len(df) / total_samples if total_samples > 0 else 0.0,
                }
                for name, df in datasets.items()
            },
            "schema": schema,
            "validation_summary": {
                "total_issues": validation_report.get("total_issues", 0),
                "issues": validation_report.get("issues", []),
            },
        }

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Exported JSON metadata: %s", dest_path.resolve())
