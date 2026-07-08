"""
engines/noise_engine.py - Noise injection engine.

Simulates real-world data collection imperfections by introducing:
    - Measurement Error: Small random perturbations to continuous features.
    - Category Flip: Random misclassifications/typos in categorical columns.
    - Missing Simulation: Randomly nullifying cell values (NaN simulation).
"""

import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class NoiseEngine:
    """
    Applies configurable noise to feature columns to mimic real-world datasets.
    """

    def __init__(self, config: GeneratorConfig, rng: Generator) -> None:
        self.config = config
        self.rng = rng

    def apply(self, builder: DatasetBuilder) -> None:
        """
        Inject noise into features inside the builder.

        Parameters
        ----------
        builder:
            DatasetBuilder instance with complete features.
        """
        logger.info("Injecting noise to simulate real-world imperfections...")

        n = builder.n_samples

        # 1. Measurement Error on Continuous Features
        # Introduce small errors on birth_weight and birth_length
        self._inject_measurement_error(builder, "birth_weight", std=0.05, clip_min=1.0, clip_max=6.0, decimals=2)
        self._inject_measurement_error(builder, "birth_length", std=0.3, clip_min=35.0, clip_max=65.0, decimals=1)

        # 2. Category Flip (Typos) on Non-Critical Categorical Features
        # Randomly flip categories with low probability (e.g. 1.0% error rate)
        self._inject_category_flip(builder, "mother_education", rate=0.01)
        self._inject_category_flip(builder, "father_education", rate=0.01)
        self._inject_category_flip(builder, "sanitation", rate=0.01)

        # 3. Missing Value Simulation (NaNs)
        # Nullify values in non-critical columns with a low rate (e.g. 1.5% missing)
        # Note: Columns configured in validation.yaml as 'no_missing_allowed'
        # (age_month, gender, birth_weight, risk_score, risk_level) MUST NOT be touched.
        self._inject_missing_simulation(builder, "mother_age", rate=0.01)
        self._inject_missing_simulation(builder, "father_working", rate=0.01)
        self._inject_missing_simulation(builder, "sanitation", rate=0.01)
        self._inject_missing_simulation(builder, "protein_intake", rate=0.01)
        self._inject_missing_simulation(builder, "exclusive_breastfeeding", rate=0.01)

        logger.info("Noise injection complete.")

    # Private Helpers

    def _inject_measurement_error(
        self,
        builder: DatasetBuilder,
        col_name: str,
        std: float,
        clip_min: float,
        clip_max: float,
        decimals: int,
    ) -> None:
        """Adds small Gaussian noise to continuous variables."""
        if not builder.has_column(col_name):
            return

        values = builder.get_column(col_name).astype(float)
        noise = self.rng.normal(0, std, len(values))
        corrupted = np.clip(values + noise, clip_min, clip_max).round(decimals)
        builder.update_column(col_name, corrupted)
        logger.debug("Injected measurement error into '%s'", col_name)

    def _inject_category_flip(
        self,
        builder: DatasetBuilder,
        col_name: str,
        rate: float,
    ) -> None:
        """Randomly flips a small fraction of categories to other valid choices."""
        if not builder.has_column(col_name) or rate <= 0:
            return

        values = builder.get_column(col_name)
        unique_cats = list(np.unique([val for val in values if val is not None]))

        if len(unique_cats) < 2:
            return

        n = len(values)
        mask = self.rng.random(n) < rate

        # Apply flips
        flipped_values = values.copy()
        for idx in np.where(mask)[0]:
            current_val = values[idx]
            # Choices excluding the current category
            choices = [c for c in unique_cats if c != current_val]
            if choices:
                flipped_values[idx] = self.rng.choice(choices)

        builder.update_column(col_name, flipped_values)
        logger.debug("Injected category flips into '%s' (flips: %d)", col_name, mask.sum())

    def _inject_missing_simulation(
        self,
        builder: DatasetBuilder,
        col_name: str,
        rate: float,
    ) -> None:
        """Simulates missing data by introducing None/NaN values."""
        if not builder.has_column(col_name) or rate <= 0:
            return

        values = builder.get_column(col_name)
        n = len(values)
        mask = self.rng.random(n) < rate

        # Convert to object array if it is not already, to allow None values
        object_values = np.array(values, dtype=object)
        object_values[mask] = None

        builder.update_column(col_name, object_values)
        logger.debug("Injected missing values into '%s' (missing: %d)", col_name, mask.sum())
