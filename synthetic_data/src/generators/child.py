"""
generators/child.py - Child feature generator.

Generates the following features for each record:
    - age_month     : child age in months (0-59)
    - gender        : biological sex (M/F)
    - birth_weight  : birth weight in kg
    - birth_length  : birth length in cm

All parameters come from distributions.yaml → child section.
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class ChildGenerator(BaseGenerator):
    """Generates child-related features based on configured distributions."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        """
        Generate child features.

        Parameters
        ----------
        n_samples:
            Number of records to generate.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: age_month, gender, birth_weight, birth_length
        """
        cfg = self._dist_config.get("child", {})

        # age_month
        age_cfg = cfg.get("age_month", {})
        age_month = self._sample_uniform_int(
            low=age_cfg.get("min", 0),
            high=age_cfg.get("max", 59),
            n=n_samples,
        )

        # gender
        gender_cfg = cfg.get("gender", {})
        gender = self._sample_categorical(
            categories=gender_cfg.get("categories", ["M", "F"]),
            probabilities=gender_cfg.get("probabilities", [0.51, 0.49]),
            n=n_samples,
        )

        # birth_weight
        bw_cfg = cfg.get("birth_weight", {})
        birth_weight = self._sample_normal_clipped(
            mean=bw_cfg.get("mean", 3.1),
            std=bw_cfg.get("std", 0.45),
            clip_min=bw_cfg.get("clip_min", 1.5),
            clip_max=bw_cfg.get("clip_max", 5.0),
            n=n_samples,
        ).round(2)

        # birth_length
        # Base: correlated with birth_weight later by RelationshipEngine.
        # Here we generate from its marginal distribution.
        bl_cfg = cfg.get("birth_length", {})
        birth_length = self._sample_normal_clipped(
            mean=bl_cfg.get("mean", 49.5),
            std=bl_cfg.get("std", 2.5),
            clip_min=bl_cfg.get("clip_min", 40.0),
            clip_max=bl_cfg.get("clip_max", 60.0),
            n=n_samples,
        ).round(1)

        logger.debug("ChildGenerator: generated %d records", n_samples)

        return {
            "age_month": age_month,
            "gender": gender,
            "birth_weight": birth_weight,
            "birth_length": birth_length,
        }
