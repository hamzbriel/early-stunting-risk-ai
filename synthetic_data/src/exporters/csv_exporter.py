"""
exporters/csv_exporter.py - Exporting train/val/test splits to CSV files.
"""

import os
import pandas as pd
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVExporter:
    """
    Exports Pandas DataFrames to structured CSV files.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.exp_config: dict[str, Any] = config.export.get("export", {})

    def export(self, datasets: dict[str, pd.DataFrame]) -> None:
        """
        Save train, validation, and test sets.

        Parameters
        ----------
        datasets:
            Mapping of split name -> DataFrame.
        """
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        files = self.exp_config.get("files", {})
        csv_options = self.exp_config.get("csv", {})
        col_order = self.exp_config.get("column_order", [])

        index = csv_options.get("index", False)
        encoding = csv_options.get("encoding", "utf-8")
        float_format = csv_options.get("float_format", "%.4f")

        for split_name, df in datasets.items():
            file_name = files.get(split_name, f"{split_name}.csv")
            dest_path = output_dir / file_name

            # Rearrange columns if ordered list provided
            ordered_cols = [col for col in col_order if col in df.columns]
            extra_cols = [col for col in df.columns if col not in ordered_cols]
            final_df = df[ordered_cols + extra_cols]

            final_df.to_csv(
                dest_path,
                index=index,
                encoding=encoding,
                float_format=float_format,
            )
            logger.info("Exported CSV: %s (%d rows)", dest_path.resolve(), len(final_df))
