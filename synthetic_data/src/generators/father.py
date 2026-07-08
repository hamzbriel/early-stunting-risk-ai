"""
generators/father.py - Father feature generator.

Generates:
    - father_education  : education level (categorical)
    - father_working    : employment status (binary)
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class FatherGenerator(BaseGenerator):
    """Generates father-related features."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        cfg = self._dist_config.get("father", {})

        # father_education
        edu_cfg = cfg.get("father_education", {})
        father_education = self._sample_categorical(
            categories=edu_cfg.get("categories", ["none", "primary", "secondary", "higher"]),
            probabilities=edu_cfg.get("probabilities", [0.08, 0.28, 0.42, 0.22]),
            n=n_samples,
        )

        # father_working
        work_cfg = cfg.get("father_working", {})
        base_p = work_cfg.get("base_probability", 0.85)
        father_working = self._sample_bernoulli(p=base_p, n=n_samples)

        logger.debug("FatherGenerator: generated %d records", n_samples)

        return {
            "father_education": father_education,
            "father_working": father_working,
        }
