"""
generators/base.py - Abstract base class for all feature generators.

All generators inherit from BaseGenerator, ensuring a consistent interface
across the pipeline. Each generator is responsible for exactly one
feature group (child, mother, father, etc.).

Usage:
    class ChildGenerator(BaseGenerator):
        def generate(self, n_samples: int) -> dict[str, np.ndarray]:
            ...
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base for all feature group generators.

    Parameters
    ----------
    config:
        Full pipeline configuration.
    rng:
        Shared NumPy random Generator.
    """

    def __init__(self, config: GeneratorConfig, rng: Generator) -> None:
        self.config = config
        self.rng = rng
        self._dist_config: dict[str, Any] = config.distributions

    @abstractmethod
    def generate(self, n_samples: int) -> dict[str, np.ndarray]:
        """
        Generate feature arrays for this group.

        Parameters
        ----------
        n_samples:
            Number of records to generate.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from feature name to array of length ``n_samples``.
        """
        ...

    # Shared helper methods available to all generators

    def _sample_categorical(
        self,
        categories: list[str],
        probabilities: list[float],
        n: int,
    ) -> np.ndarray:
        """
        Sample from a categorical distribution.

        Parameters
        ----------
        categories:
            List of category labels.
        probabilities:
            Corresponding probabilities (must sum to 1.0).
        n:
            Number of samples.
        """
        probs = np.array(probabilities, dtype=float)
        probs /= probs.sum()  # normalize to handle floating point drift
        return self.rng.choice(categories, size=n, p=probs)

    def _sample_normal_clipped(
        self,
        mean: float,
        std: float,
        clip_min: float,
        clip_max: float,
        n: int,
    ) -> np.ndarray:
        """Sample from a normal distribution, clipped to [clip_min, clip_max]."""
        values = self.rng.normal(loc=mean, scale=std, size=n)
        return np.clip(values, clip_min, clip_max)

    def _sample_bernoulli(self, p: float, n: int) -> np.ndarray:
        """
        Sample binary (0/1) values from a Bernoulli distribution.

        Parameters
        ----------
        p:
            Probability of 1 (True).
        """
        return self.rng.binomial(n=1, p=np.clip(p, 0.0, 1.0), size=n)

    def _sample_uniform_int(self, low: int, high: int, n: int) -> np.ndarray:
        """Sample integers uniformly from [low, high] inclusive."""
        return self.rng.integers(low=low, high=high + 1, size=n)
