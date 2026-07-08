"""
tests/unit/test_generators.py - Unit tests for the synthetic data generators and engines.
"""

import pytest
import numpy as np
import pandas as pd

from synthetic_data.src.core.config_loader import load_config
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.core.random_manager import RandomManager
from synthetic_data.src.generators.child import ChildGenerator
from synthetic_data.src.engines.relationship_engine import RelationshipEngine
from synthetic_data.src.engines.risk_engine import RiskEngine
from synthetic_data.src.validators.range_validator import RangeValidator


def test_config_loader(sample_config_path):
    """Test that configurations load and validate successfully."""
    config = load_config(config_dir=sample_config_path)
    assert config is not None
    assert config.generator.n_samples > 0
    assert "child" in config.distributions
    assert "edges" in config.relationships


def test_child_generator(sample_config_path):
    """Test that ChildGenerator produces requested shapes and valid ranges."""
    config = load_config(config_dir=sample_config_path)
    rng = np.random.default_rng(seed=42)
    gen = ChildGenerator(config, rng)

    n_samples = 100
    features = gen.generate(n_samples)

    assert len(features["age_month"]) == n_samples
    assert len(features["gender"]) == n_samples
    assert len(features["birth_weight"]) == n_samples
    assert len(features["birth_length"]) == n_samples

    # Check bounds
    assert np.all(features["age_month"] >= 0)
    assert np.all(features["age_month"] <= 59)
    assert set(features["gender"]).issubset({"M", "F"})


def test_dataset_builder():
    """Test that DatasetBuilder accumulates columns properly."""
    builder = DatasetBuilder(n_samples=10)
    builder.add_column("feat_a", np.ones(10))
    builder.add_column("feat_b", np.zeros(10))

    assert builder.shape == (10, 2)
    assert builder.has_column("feat_a")

    df = builder.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 2)
