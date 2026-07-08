"""
engines/relationship_engine.py - Core dependency relationship modeling.

Applies Directed Acyclic Graph (DAG) conditional probability shifts and
correlations to ensure that generated features are not statistically
independent but reflect realistic health and socioeconomic patterns.

Specifically models:
    - Mother education -> Mother working -> Family income
    - Family income -> House density -> Sanitation -> Clean water
    - Clean water + Sanitation -> Diarrhea history
    - Family income + Mother education -> Protein intake + Vitamin intake
    - Healthcare access -> Immunization -> Diarrhea history
    - Birth weight -> Birth length (correlated)
"""

import numpy as np
import pandas as pd
from numpy.random import Generator
from typing import Any

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class RelationshipEngine:
    """
    Applies conditional probability shifts and correlations to features
    based on configurations defined in relationships.yaml.
    """

    def __init__(self, config: GeneratorConfig, rng: Generator) -> None:
        self.config = config
        self.rng = rng
        self.rel_config: dict[str, Any] = config.relationships

    def apply(self, builder: DatasetBuilder) -> None:
        """
        Modify the generated base features in-place inside the builder
        to reflect dependencies.

        Parameters
        ----------
        builder:
            DatasetBuilder instance with base marginal features already loaded.
        """
        logger.info("Applying inter-feature relationships...")

        # We will process relationships based on the processing order
        # to ensure dependencies are resolved step-by-step.
        # But we will implement the specific rules directly to guarantee correctness
        # and precision, matching the DAG spec.

        n = builder.n_samples

        # 1. Mother Education -> Mother Working 
        # mother_education is categorical: "none", "primary", "secondary", "higher"
        # mother_working is binary (0/1)
        self._apply_mother_education_to_working(builder, n)

        # 2. Mother Education + Father Education -> Family Income 
        # family_income is categorical: "very_low", "low", "medium", "high"
        self._apply_parents_education_to_income(builder, n)

        # 3. Family Income -> House Density 
        # house_density is categorical: "low", "medium", "high"
        self._apply_income_to_house_density(builder, n)

        # 4. House Density + Family Income -> Sanitation 
        # sanitation is categorical: "poor", "fair", "good"
        self._apply_density_and_income_to_sanitation(builder, n)

        # 5. Sanitation + Family Income -> Clean Water 
        # clean_water is binary (0/1)
        self._apply_sanitation_and_income_to_clean_water(builder, n)

        # 6. Family Income + Mother Education -> Healthcare Access 
        # healthcare_access is categorical: "poor", "fair", "good"
        self._apply_income_and_education_to_healthcare_access(builder, n)

        # 7. Healthcare Access -> Immunization 
        # immunization is categorical: "none", "partial", "complete"
        self._apply_healthcare_to_immunization(builder, n)

        # 8. Clean Water + Sanitation + Immunization -> Diarrhea History
        # diarrhea_history is binary (0/1)
        self._apply_wash_and_imm_to_diarrhea(builder, n)

        # 9. Family Income + Mother Education -> Protein & Vitamin Intake
        # protein_intake, vitamin_intake: categorical: "low", "medium", "high"
        self._apply_income_and_education_to_nutrition(builder, n)

        # 10. Birth Weight -> Birth Length (Correlation) 
        # Apply a correlation of ~0.75 between birth_weight and birth_length
        self._apply_birth_weight_to_length_correlation(builder, n)

        logger.info("Relationship Engine applied successfully.")

    # Rule Implementations

    def _apply_mother_education_to_working(self, builder: DatasetBuilder, n: int) -> None:
        edu = builder.get_column("mother_education")
        working = builder.get_column("mother_working")

        # Get baseline probabilities
        base_p = self.config.distributions.get("mother", {}).get("mother_working", {}).get("base_probability", 0.40)

        # Define shifts from relationships.yaml or code hardcoded logic mirroring config
        # none: -0.20, primary: -0.05, secondary: +0.10, higher: +0.25
        probs = np.full(n, base_p)
        probs[edu == "none"] += -0.20
        probs[edu == "primary"] += -0.05
        probs[edu == "secondary"] += 0.10
        probs[edu == "higher"] += 0.25

        probs = np.clip(probs, 0.02, 0.98) # Keep within realistic bounds
        new_working = self.rng.binomial(1, probs)
        builder.update_column("mother_working", new_working)

    def _apply_parents_education_to_income(self, builder: DatasetBuilder, n: int) -> None:
        m_edu = builder.get_column("mother_education")
        f_edu = builder.get_column("father_education")

        # family_income categories
        categories = ["very_low", "low", "medium", "high"]
        # Default distribution: [0.20, 0.35, 0.30, 0.15]
        base_probs = np.array([0.20, 0.35, 0.30, 0.15])

        new_income = []
        for i in range(n):
            me = m_edu[i]
            fe = f_edu[i]
            probs = base_probs.copy()

            # Apply shifts based on parental education levels
            if me == "higher" and fe == "higher":
                probs = np.array([0.02, 0.08, 0.30, 0.60])
            elif me in ("secondary", "higher") or fe in ("secondary", "higher"):
                probs = np.array([0.05, 0.25, 0.50, 0.20])
            elif me == "primary" and fe == "primary":
                probs = np.array([0.20, 0.50, 0.25, 0.05])
            elif me == "none" and fe == "none":
                probs = np.array([0.65, 0.25, 0.08, 0.02])
            else:
                # Moderate mix
                probs = np.array([0.30, 0.40, 0.22, 0.08])

            probs /= probs.sum()
            val = self.rng.choice(categories, p=probs)
            new_income.append(val)

        builder.update_column("family_income", np.array(new_income))

    def _apply_income_to_house_density(self, builder: DatasetBuilder, n: int) -> None:
        income = builder.get_column("family_income")
        categories = ["low", "medium", "high"]
        # Default distribution: [0.30, 0.45, 0.25]
        base_probs = np.array([0.30, 0.45, 0.25])

        new_density = []
        for i in range(n):
            inc = income[i]
            probs = base_probs.copy()
            if inc == "high":
                probs = np.array([0.65, 0.25, 0.10])
            elif inc == "medium":
                probs = np.array([0.40, 0.45, 0.15])
            elif inc == "low":
                probs = np.array([0.20, 0.50, 0.30])
            elif inc == "very_low":
                probs = np.array([0.08, 0.37, 0.55])

            probs /= probs.sum()
            new_density.append(self.rng.choice(categories, p=probs))

        builder.update_column("house_density", np.array(new_density))

    def _apply_density_and_income_to_sanitation(self, builder: DatasetBuilder, n: int) -> None:
        density = builder.get_column("house_density")
        income = builder.get_column("family_income")
        categories = ["poor", "fair", "good"]

        new_sanitation = []
        for i in range(n):
            dens = density[i]
            inc = income[i]

            # Combine signals
            # Base distribution: [0.25, 0.40, 0.35]
            if inc == "high" and dens == "low":
                probs = np.array([0.03, 0.22, 0.75])
            elif inc == "very_low" and dens == "high":
                probs = np.array([0.70, 0.25, 0.05])
            elif inc in ("low", "very_low") or dens == "high":
                probs = np.array([0.45, 0.45, 0.10])
            elif inc == "high" or dens == "low":
                probs = np.array([0.08, 0.32, 0.60])
            else:
                probs = np.array([0.25, 0.50, 0.25])

            probs /= probs.sum()
            new_sanitation.append(self.rng.choice(categories, p=probs))

        builder.update_column("sanitation", np.array(new_sanitation))

    def _apply_sanitation_and_income_to_clean_water(self, builder: DatasetBuilder, n: int) -> None:
        sanitation = builder.get_column("sanitation")
        income = builder.get_column("family_income")

        # Base clean_water probability: 0.65
        probs = np.full(n, 0.65)

        # Adjust
        probs[sanitation == "poor"] -= 0.20
        probs[sanitation == "good"] += 0.20
        probs[income == "very_low"] -= 0.15
        probs[income == "high"] += 0.15

        probs = np.clip(probs, 0.05, 0.95)
        new_cw = self.rng.binomial(1, probs)
        builder.update_column("clean_water", new_cw)

    def _apply_income_and_education_to_healthcare_access(self, builder: DatasetBuilder, n: int) -> None:
        income = builder.get_column("family_income")
        m_edu = builder.get_column("mother_education")
        categories = ["poor", "fair", "good"]

        new_access = []
        for i in range(n):
            inc = income[i]
            me = m_edu[i]

            # Base distribution: [0.20, 0.40, 0.40]
            if inc == "high" and me == "higher":
                probs = np.array([0.02, 0.18, 0.80])
            elif inc == "very_low" and me == "none":
                probs = np.array([0.65, 0.30, 0.05])
            elif inc in ("very_low", "low"):
                probs = np.array([0.45, 0.40, 0.15])
            elif inc == "high" or me == "higher":
                probs = np.array([0.05, 0.25, 0.70])
            else:
                probs = np.array([0.15, 0.45, 0.40])

            probs /= probs.sum()
            new_access.append(self.rng.choice(categories, p=probs))

        builder.update_column("healthcare_access", np.array(new_access))

    def _apply_healthcare_to_immunization(self, builder: DatasetBuilder, n: int) -> None:
        access = builder.get_column("healthcare_access")
        categories = ["none", "partial", "complete"]

        new_imm = []
        for i in range(n):
            acc = access[i]

            # Base: [0.10, 0.30, 0.60]
            if acc == "good":
                probs = np.array([0.02, 0.13, 0.85])
            elif acc == "fair":
                probs = np.array([0.08, 0.32, 0.60])
            else: # poor
                probs = np.array([0.35, 0.45, 0.20])

            probs /= probs.sum()
            new_imm.append(self.rng.choice(categories, p=probs))

        builder.update_column("immunization", np.array(new_imm))

    def _apply_wash_and_imm_to_diarrhea(self, builder: DatasetBuilder, n: int) -> None:
        cw = builder.get_column("clean_water")
        san = builder.get_column("sanitation")
        imm = builder.get_column("immunization")

        # Base diarrhea_history probability: 0.30
        probs = np.full(n, 0.30)

        # Adjust based on sanitation and clean water (WASH)
        # sanitation: poor, fair, good
        # clean_water: 0, 1
        # immunization: none, partial, complete

        for i in range(n):
            # WASH risk
            if cw[i] == 0:
                probs[i] += 0.15
            if san[i] == "poor":
                probs[i] += 0.20
            elif san[i] == "good":
                probs[i] -= 0.15

            # Immunization protection (rotavirus, measles prevent general health issues)
            if imm[i] == "none":
                probs[i] += 0.10
            elif imm[i] == "complete":
                probs[i] -= 0.10

        probs = np.clip(probs, 0.03, 0.85)
        new_diarrhea = self.rng.binomial(1, probs)
        builder.update_column("diarrhea_history", new_diarrhea)

    def _apply_income_and_education_to_nutrition(self, builder: DatasetBuilder, n: int) -> None:
        income = builder.get_column("family_income")
        m_edu = builder.get_column("mother_education")
        categories = ["low", "medium", "high"]

        new_protein = []
        new_vitamin = []

        for i in range(n):
            inc = income[i]
            me = m_edu[i]

            # Base protein distribution: [0.30, 0.45, 0.25]
            # Base vitamin distribution: [0.25, 0.45, 0.30]

            # Protein intake adjustments
            if inc == "high" and me == "higher":
                p_probs = np.array([0.03, 0.22, 0.75])
            elif inc == "very_low":
                p_probs = np.array([0.65, 0.30, 0.05])
            elif inc == "low":
                p_probs = np.array([0.45, 0.45, 0.10])
            elif inc == "high" or me == "higher":
                p_probs = np.array([0.08, 0.37, 0.55])
            else:
                p_probs = np.array([0.25, 0.50, 0.25])

            # Vitamin intake adjustments (similar but slightly different)
            if inc == "high" and me == "higher":
                v_probs = np.array([0.02, 0.23, 0.75])
            elif inc == "very_low":
                v_probs = np.array([0.55, 0.38, 0.07])
            elif inc == "low":
                v_probs = np.array([0.38, 0.50, 0.12])
            elif inc == "high" or me == "higher":
                v_probs = np.array([0.07, 0.38, 0.55])
            else:
                v_probs = np.array([0.20, 0.50, 0.30])

            p_probs /= p_probs.sum()
            v_probs /= v_probs.sum()

            new_protein.append(self.rng.choice(categories, p=p_probs))
            new_vitamin.append(self.rng.choice(categories, p=v_probs))

        builder.update_column("protein_intake", np.array(new_protein))
        builder.update_column("vitamin_intake", np.array(new_vitamin))

    def _apply_birth_weight_to_length_correlation(self, builder: DatasetBuilder, n: int) -> None:
        bw = builder.get_column("birth_weight")

        # We want to adjust birth_length to have a ~0.75 correlation with birth_weight
        # Formula: length_z = rho * weight_z + sqrt(1 - rho^2) * noise_z
        # Let's convert birth_weight to z-scores, generate correlated z-scores,
        # and then map back to birth_length distribution.

        bw_mean = bw.mean()
        bw_std = bw.std()

        # Prevent division by zero
        if bw_std == 0:
            bw_std = 0.45

        bw_z = (bw - bw_mean) / bw_std

        # Generate correlated z-scores
        rho = 0.75
        noise_z = self.rng.normal(0, 1, n)
        bl_z = rho * bw_z + np.sqrt(1 - rho**2) * noise_z

        # Map back to birth_length distribution
        # distribution.yaml -> birth_length -> mean: 49.5, std: 2.5, clip: [40, 60]
        bl_cfg = self.config.distributions.get("child", {}).get("birth_length", {})
        mean = bl_cfg.get("mean", 49.5)
        std = bl_cfg.get("std", 2.5)
        clip_min = bl_cfg.get("clip_min", 40.0)
        clip_max = bl_cfg.get("clip_max", 60.0)

        new_bl = (bl_z * std) + mean
        new_bl = np.clip(new_bl, clip_min, clip_max).round(1)

        builder.update_column("birth_length", new_bl)
