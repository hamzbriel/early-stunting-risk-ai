"""
engines/risk_engine.py - Risk calculation and labeling engine.

Computes the `risk_score` for each child using a weighted scoring model
augmented by interaction effects and standard Gaussian noise.
Converts the final score into a categorical `risk_level` (Low/Medium/High).

Design features:
    - Weighted contribution of 10 key health/socioeconomic features.
    - Interaction effects representing compound risk or protective synergies.
    - Additive Gaussian noise to simulate real-world variance.
    - Threshold-based labeling.
"""

import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskEngine:
    """
    Computes risk score and risk level for child stunting based on
    predefined weights, scoring maps, interactions, and thresholds.
    """

    def __init__(self, config: GeneratorConfig, rng: Generator) -> None:
        self.config = config
        self.rng = rng
        self.risk_config: dict[str, Any] = config.risk_rules

    def apply(self, builder: DatasetBuilder) -> None:
        """
        Compute risk_score and risk_level for the dataset in builder
        and add these columns.

        Parameters
        ----------
        builder:
            DatasetBuilder instance with base and related features.
        """
        logger.info("Computing risk scores...")

        n = builder.n_samples
        weights = self.risk_config.get("weights", {})
        scoring = self.risk_config.get("scoring", {})
        interactions = self.risk_config.get("interactions", [])
        thresholds = self.risk_config.get("thresholds", {})
        noise_cfg = self.risk_config.get("risk_noise", {})

        # Extract features needed for risk scoring
        bw = builder.get_column("birth_weight")
        bl = builder.get_column("birth_length")
        protein = builder.get_column("protein_intake")
        income = builder.get_column("family_income")
        sanitation = builder.get_column("sanitation")
        m_edu = builder.get_column("mother_education")
        cw = builder.get_column("clean_water")
        diarrhea = builder.get_column("diarrhea_history")
        immunization = builder.get_column("immunization")
        healthcare = builder.get_column("healthcare_access")

        # 1. Calculate base feature risk scores (0 to 100 scale per feature)
        # Continuous: Inverse normalization (lower values = higher risk)
        score_bw = self._score_continuous_inverse(bw, scoring["birth_weight"])
        score_bl = self._score_continuous_inverse(bl, scoring["birth_length"])

        # Categorical / Binary
        score_protein = self._score_categorical(protein, scoring["protein_intake"])
        score_income = self._score_categorical(income, scoring["family_income"])
        score_sanitation = self._score_categorical(sanitation, scoring["sanitation"])
        score_m_edu = self._score_categorical(m_edu, scoring["mother_education"])
        score_cw = self._score_categorical(cw, scoring["clean_water"])
        score_diarrhea = self._score_categorical(diarrhea, scoring["diarrhea_history"])
        score_immunization = self._score_categorical(immunization, scoring["immunization"])
        score_healthcare = self._score_categorical(healthcare, scoring["healthcare_access"])

        # 2. Weighted Sum of Scores
        weighted_score = (
            weights["birth_weight"] * score_bw
            + weights["protein_intake"] * score_protein
            + weights["family_income"] * score_income
            + weights["sanitation"] * score_sanitation
            + weights["mother_education"] * score_m_edu
            + weights["clean_water"] * score_cw
            + weights["diarrhea_history"] * score_diarrhea
            + weights["immunization"] * score_immunization
            + weights["birth_length"] * score_bl
            + weights["healthcare_access"] * score_healthcare
        )

        # 3. Apply Interaction Effects
        interaction_effects = np.zeros(n)
        for i in range(n):
            bonus = 0.0

            # severe_poverty_poor_sanitation: +8.0
            if income[i] == "very_low" and sanitation[i] == "poor":
                bonus += 8.0

            # low_birth_weight_low_protein: +7.0
            if bw[i] < 2.5 and protein[i] == "low":
                bonus += 7.0

            # no_clean_water_diarrhea: +6.0
            if cw[i] == 0 and diarrhea[i] == 1:
                bonus += 6.0

            # uneducated_mother_low_income: +5.0
            if m_edu[i] == "none" and income[i] == "very_low":
                bonus += 5.0

            # educated_mother_high_income: -7.0
            if m_edu[i] == "higher" and income[i] == "high":
                bonus -= 7.0

            # good_sanitation_clean_water_immunized: -6.0
            if sanitation[i] == "good" and cw[i] == 1 and immunization[i] == "complete":
                bonus -= 6.0

            interaction_effects[i] = bonus

        # Combine
        risk_score = weighted_score + interaction_effects

        # 4. Add Small Gaussian Noise
        noise_std = noise_cfg.get("gaussian_std", 3.5)
        if noise_std > 0:
            risk_score += self.rng.normal(0, noise_std, n)

        # Clip final score to [0, 100]
        risk_score = np.clip(risk_score, 0.0, 100.0).round(2)

        # 5. Convert to Risk Level
        risk_level = []
        low_max = thresholds.get("low", {}).get("max", 35)
        med_max = thresholds.get("medium", {}).get("max", 65)

        for score in risk_score:
            if score <= low_max:
                risk_level.append("Low")
            elif score <= med_max:
                risk_level.append("Medium")
            else:
                risk_level.append("High")

        # 6. Save to Builder
        builder.add_column("risk_score", risk_score)
        builder.add_column("risk_level", np.array(risk_level))

        logger.info(
            "Risk Engine complete. Summary of risk levels: Low=%d, Medium=%d, High=%d",
            sum(1 for x in risk_level if x == "Low"),
            sum(1 for x in risk_level if x == "Medium"),
            sum(1 for x in risk_level if x == "High"),
        )

    # Helper Methods

    def _score_continuous_inverse(self, values: np.ndarray, config: dict) -> np.ndarray:
        """
        Maps continuous values to risk score inverse-linearly.
        Lower value = higher risk score.
        """
        ref_min = config["reference_min"]
        ref_max = config["reference_max"]
        risk_min = config["risk_at_min"] # risk score at reference_min
        risk_max = config["risk_at_max"] # risk score at reference_max

        # Clip values to references to avoid extrapolation
        clipped_vals = np.clip(values, ref_min, ref_max)

        # Linear interpolation
        # val = ref_min -> risk_min
        # val = ref_max -> risk_max
        factor = (clipped_vals - ref_min) / (ref_max - ref_min)
        scores = risk_min + factor * (risk_max - risk_min)
        return scores

    def _score_categorical(self, values: np.ndarray, config: dict) -> np.ndarray:
        """
        Maps categorical values to risk score based on custom configuration mapping.
        """
        mapping = config["mapping"]
        # Convert mapping keys to strings/ints as appropriate
        # Pydantic loads keys as is, but let's handle string representation of integers for safety
        string_map = {str(k): v for k, v in mapping.items()}

        scores = np.zeros(len(values))
        for i, val in enumerate(values):
            # Check string representation first
            scores[i] = string_map.get(str(val), 50.0) # default to mid-point if missing
        return scores
