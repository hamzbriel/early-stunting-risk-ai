"""
core/random_manager.py - Random number generator management for the pipeline.

Thin wrapper that bridges the GeneratorConfig (seed) with the
utils/seed.py module. The pipeline calls this once at startup.
"""

from numpy.random import Generator

from synthetic_data.src.core.config_loader import GeneratorConfig
from synthetic_data.src.utils.logger import get_logger
from synthetic_data.src.utils.seed import fork_rng, get_rng, init_rng

logger = get_logger(__name__)


class RandomManager:
    """
    Manages the global NumPy random Generator for the pipeline.

    Ensures a single seed is used across all generators and engines,
    providing full reproducibility.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._seed: int = config.generator.seed
        self._rng: Generator | None = None

    def initialize(self) -> Generator:
        """Initialize the global RNG with the configured seed."""
        self._rng = init_rng(seed=self._seed)
        logger.info("Random Generator initialized with seed=%d", self._seed)
        return self._rng

    @property
    def rng(self) -> Generator:
        """Return the active global Generator. Raises if not initialized."""
        if self._rng is None:
            raise RuntimeError("Call RandomManager.initialize() first.")
        return get_rng()

    def fork(self, offset: int = 0) -> Generator:
        """
        Create an independent child Generator for isolated use (e.g., testing).

        Parameters
        ----------
        offset:
            Added to the global seed to produce a distinct stream.
        """
        return fork_rng(child_seed_offset=offset)
