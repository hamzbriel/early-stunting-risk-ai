"""
tests/unit/test_engines.py - Unit tests for the Relationship and Risk Engines.
"""

import pytest
import numpy as np
import pandas as pd

from synthetic_data.src.core.config_loader import load_config
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.generators.child import ChildGenerator
from synthetic_data.src.generators.mother import MotherGenerator
from synthetic_data.src.generators.father import FatherGenerator
from synthetic_data.src.generators.household import HouseholdGenerator
from synthetic_data.src.generators.nutrition import NutritionGenerator
from synthetic_data.src.generators.healthcare import HealthcareGenerator
from synthetic_data.src.engines.relationship_engine import RelationshipEngine
from synthetic_data.src.engines.risk_engine import RiskEngine


def test_engines_pipeline(sample_config_path):
    """Test the complete generation, relationship, and risk pipeline on a small scale."""
    config = load_config(config_dir=sample_config_path)
    rng = np.random.default_rng(seed=1337)
    n_samples = 50

    # Build features
    builder = DatasetBuilder(n_samples=n_samples)

    generators = [
        ChildGenerator(config, rng),
        MotherGenerator(config, rng),
        FatherGenerator(config, rng),
        HouseholdGenerator(config, rng),
        NutritionGenerator(config, rng),
        HealthcareGenerator(config, rng),
    ]

    for gen in generators:
        builder.add_columns(gen.generate(n_samples))

    # Apply Relationship Engine
    rel_engine = RelationshipEngine(config, rng)
    rel_engine.apply(builder)

    # Verify correlations are applied (e.g. birth_weight to birth_length correlation is positive)
    bw = builder.get_column("birth_weight")
    bl = builder.get_column("birth_length")
    corr = np.corrcoef(bw, bl)[0, 1]
    assert corr > 0.4  # Strong positive correlation expected

    # Apply Risk Engine
    risk_engine = RiskEngine(config, rng)
    risk_engine.apply(builder)

    assert builder.has_column("risk_score")
    assert builder.has_column("risk_level")

    scores = builder.get_column("risk_score")
    levels = builder.get_column("risk_level")

    assert len(scores) == n_samples
    assert len(levels) == n_samples
    assert np.all(scores >= 0) and np.all(scores <= 100)
    assert set(levels).issubset({"Low", "Medium", "High"})
