"""
exporters/dictionary_exporter.py - Exporting the dataset's Data Dictionary to CSV.
"""

import os
import pandas as pd
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class DictionaryExporter:
    """
    Builds and exports a data dictionary explaining each column, type, description, and source.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.exp_config: dict[str, Any] = config.export.get("export", {})

    def export(self, df: pd.DataFrame) -> None:
        """
        Extract descriptions and types for all columns, and export to CSV.
        """
        output_dir = self.config.output_dir
        file_name = self.exp_config.get("files", {}).get("data_dictionary", "data_dictionary.csv")
        dest_path = output_dir / file_name

        # Mapping of all features to their category and description from distributions.yaml and risk_rules.yaml
        dists = self.config.distributions
        descriptions = {}

        # Scan distributions.yaml categories
        for category_key in ["child", "mother", "father", "household", "nutrition", "healthcare"]:
            feat_group = dists.get(category_key, {})
            for feat_name, feat_cfg in feat_group.items():
                descriptions[feat_name] = {
                    "group": category_key,
                    "description": feat_cfg.get("description", ""),
                    "values_range": str(feat_cfg.get("categories", "")) or f"{feat_cfg.get('min', '')}-{feat_cfg.get('max', '')}",
                }

        # Add target features manually
        descriptions["risk_score"] = {
            "group": "target",
            "description": "Calculated early stunting risk score (0-100)",
            "values_range": "0.0 - 100.0",
        }
        descriptions["risk_level"] = {
            "group": "target",
            "description": "Target stunting risk classification tier",
            "values_range": "['Low', 'Medium', 'High']",
        }

        # Build records
        records = []
        for col in df.columns:
            meta = descriptions.get(col, {"group": "unknown", "description": "", "values_range": ""})
            records.append({
                "Feature Name": col,
                "Group": meta["group"].upper(),
                "Data Type": str(df[col].dtype),
                "Description": meta["description"],
                "Values Range / Categories": meta["values_range"],
            })

        dict_df = pd.DataFrame(records)
        dict_df.to_csv(dest_path, index=False)

        logger.info("Exported CSV data dictionary: %s", dest_path.resolve())
