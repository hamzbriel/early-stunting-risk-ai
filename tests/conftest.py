"""
Shared test fixtures and configuration for the entire test suite.
This conftest.py is loaded automatically by pytest before running any tests.
"""

import pytest
import numpy as np
import pandas as pd


# Fixtures - Random number generator (seeded for reproducibility)

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded NumPy random Generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def sample_config_path() -> str:
    """Path to the synthetic data config directory."""
    import os
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "synthetic_data",
        "config",
    )


@pytest.fixture(scope="function")
def small_dataframe() -> pd.DataFrame:
    """A minimal 10-row DataFrame for unit testing validators."""
    return pd.DataFrame(
        {
            "age_month": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "birth_weight": [2.5, 3.0, 3.5, 2.8, 3.2, 2.6, 3.1, 2.9, 3.4, 3.0],
            "birth_length": [48.0, 50.0, 51.0, 49.0, 52.0, 47.0, 50.5, 49.5, 51.5, 50.0],
            "risk_score": [25.0, 45.0, 70.0, 35.0, 60.0, 80.0, 20.0, 55.0, 40.0, 65.0],
            "risk_level": ["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium", "Low", "Medium"],
        }
    )
