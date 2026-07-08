"""
generators/nutrition.py - Nutrition feature generator.

Generates initial (marginal) values for:
    - exclusive_breastfeeding  : 6-month exclusive breastfeeding (binary)
    - protein_intake           : protein adequacy (categorical)
    - vitamin_intake           : micronutrient adequacy (categorical)

RelationshipEngine will adjust protein_intake and vitamin_intake
based on family_income and mother_education.
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class NutritionGenerator(BaseGenerator):
    """Generates nutrition-related features."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        cfg = self._dist_config.get("nutrition", {})

        # exclusive_breastfeeding
        bf_cfg = cfg.get("exclusive_breastfeeding", {})
        exclusive_breastfeeding = self._sample_bernoulli(
            p=bf_cfg.get("base_probability", 0.55),
            n=n_samples,
        )

        # protein_intake
        prot_cfg = cfg.get("protein_intake", {})
        protein_intake = self._sample_categorical(
            categories=prot_cfg.get("categories", ["low", "medium", "high"]),
            probabilities=prot_cfg.get("probabilities", [0.30, 0.45, 0.25]),
            n=n_samples,
        )

        # vitamin_intake
        vit_cfg = cfg.get("vitamin_intake", {})
        vitamin_intake = self._sample_categorical(
            categories=vit_cfg.get("categories", ["low", "medium", "high"]),
            probabilities=vit_cfg.get("probabilities", [0.25, 0.45, 0.30]),
            n=n_samples,
        )

        logger.debug("NutritionGenerator: generated %d records", n_samples)

        return {
            "exclusive_breastfeeding": exclusive_breastfeeding,
            "protein_intake": protein_intake,
            "vitamin_intake": vitamin_intake,
        }
