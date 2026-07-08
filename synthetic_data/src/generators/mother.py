"""
generators/mother.py - Mother feature generator.

Generates:
    - mother_age        : age in years
    - mother_education  : education level (categorical)
    - mother_working    : employment status (binary)

Note: mother_working base probability is generated here; the
RelationshipEngine will adjust it based on mother_education.
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class MotherGenerator(BaseGenerator):
    """Generates mother-related features."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        cfg = self._dist_config.get("mother", {})

        # mother_age
        age_cfg = cfg.get("mother_age", {})
        mother_age = self._sample_normal_clipped(
            mean=age_cfg.get("mean", 28.0),
            std=age_cfg.get("std", 6.0),
            clip_min=age_cfg.get("clip_min", 15),
            clip_max=age_cfg.get("clip_max", 50),
            n=n_samples,
        ).round(0).astype(int)

        # mother_education
        edu_cfg = cfg.get("mother_education", {})
        mother_education = self._sample_categorical(
            categories=edu_cfg.get("categories", ["none", "primary", "secondary", "higher"]),
            probabilities=edu_cfg.get("probabilities", [0.10, 0.30, 0.40, 0.20]),
            n=n_samples,
        )

        # mother_working
        # Base probability - will be fine-tuned by RelationshipEngine
        work_cfg = cfg.get("mother_working", {})
        base_p = work_cfg.get("base_probability", 0.40)
        mother_working = self._sample_bernoulli(p=base_p, n=n_samples)

        logger.debug("MotherGenerator: generated %d records", n_samples)

        return {
            "mother_age": mother_age,
            "mother_education": mother_education,
            "mother_working": mother_working,
        }
