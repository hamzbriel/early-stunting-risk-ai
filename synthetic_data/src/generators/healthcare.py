"""
generators/healthcare.py - Healthcare feature generator.

Generates initial (marginal) values for:
    - immunization        : immunization completeness (categorical)
    - diarrhea_history    : recent diarrhea history (binary)
    - healthcare_access   : facility access quality (categorical)

RelationshipEngine will adjust these based on:
    - healthcare_access → immunization
    - clean_water + sanitation + immunization → diarrhea_history
"""

import numpy as np

from synthetic_data.src.generators.base import BaseGenerator
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class HealthcareGenerator(BaseGenerator):
    """Generates healthcare-related features."""

    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        cfg = self._dist_config.get("healthcare", {})

        # healthcare_access
        access_cfg = cfg.get("healthcare_access", {})
        healthcare_access = self._sample_categorical(
            categories=access_cfg.get("categories", ["poor", "fair", "good"]),
            probabilities=access_cfg.get("probabilities", [0.20, 0.40, 0.40]),
            n=n_samples,
        )

        # immunization
        imm_cfg = cfg.get("immunization", {})
        immunization = self._sample_categorical(
            categories=imm_cfg.get("categories", ["none", "partial", "complete"]),
            probabilities=imm_cfg.get("probabilities", [0.10, 0.30, 0.60]),
            n=n_samples,
        )

        # diarrhea_history
        diag_cfg = cfg.get("diarrhea_history", {})
        diarrhea_history = self._sample_bernoulli(
            p=diag_cfg.get("base_probability", 0.30),
            n=n_samples,
        )

        logger.debug("HealthcareGenerator: generated %d records", n_samples)

        return {
            "healthcare_access": healthcare_access,
            "immunization": immunization,
            "diarrhea_history": diarrhea_history,
        }
