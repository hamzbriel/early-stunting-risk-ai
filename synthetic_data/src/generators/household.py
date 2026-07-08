"""
generators/household.py - Household feature generator.

Generates initial (marginal) values for:
    - family_income  : income tier (categorical)
    - sanitation     : sanitation quality (categorical)
    - clean_water    : access to clean water (binary)
    - electricity    : access to electricity (binary)
    - house_density  : occupancy density (categorical)

Important: These are base (marginal) values. The RelationshipEngine
will adjust them based on parental education and income dependencies.
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class HouseholdGenerator(BaseGenerator):
    """Generates household-related features."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        cfg = self._dist_config.get("household", {})

        # family_income
        inc_cfg = cfg.get("family_income", {})
        family_income = self._sample_categorical(
            categories=inc_cfg.get("categories", ["very_low", "low", "medium", "high"]),
            probabilities=inc_cfg.get("probabilities", [0.20, 0.35, 0.30, 0.15]),
            n=n_samples,
        )

        # sanitation
        san_cfg = cfg.get("sanitation", {})
        sanitation = self._sample_categorical(
            categories=san_cfg.get("categories", ["poor", "fair", "good"]),
            probabilities=san_cfg.get("probabilities", [0.25, 0.40, 0.35]),
            n=n_samples,
        )

        # clean_water
        cw_cfg = cfg.get("clean_water", {})
        clean_water = self._sample_bernoulli(
            p=cw_cfg.get("base_probability", 0.65),
            n=n_samples,
        )

        # electricity
        elec_cfg = cfg.get("electricity", {})
        electricity = self._sample_bernoulli(
            p=elec_cfg.get("base_probability", 0.85),
            n=n_samples,
        )

        # house_density
        hd_cfg = cfg.get("house_density", {})
        house_density = self._sample_categorical(
            categories=hd_cfg.get("categories", ["low", "medium", "high"]),
            probabilities=hd_cfg.get("probabilities", [0.30, 0.45, 0.25]),
            n=n_samples,
        )

        logger.debug("HouseholdGenerator: generated %d records", n_samples)

        return {
            "family_income": family_income,
            "sanitation": sanitation,
            "clean_water": clean_water,
            "electricity": electricity,
            "house_density": house_density,
        }
