"""
utils/seed.py - Centralized random seed management.

Provides a single NumPy random Generator instance shared across
all pipeline components. This ensures full reproducibility of
the generated dataset given the same seed.

Design note:
    We use numpy.random.Generator (the modern API) rather than
    numpy.random.RandomState or the legacy numpy.random.seed() function.
    Generator instances can be passed between functions, making the
    randomness flow explicit and testable.

Usage:
    from synthetic_data.src.utils.seed import get_rng, init_rng

    # In main.py / pipeline.py - initialize once
    init_rng(seed=42)

    # In any generator / engine - retrieve the shared instance
    rng = get_rng()
    values = rng.normal(loc=3.1, scale=0.45, size=1000)
"""

import numpy as np
from numpy.random import Generator

# Module-level private state
_rng: Generator | None = None
_seed: int | None = None


def init_rng(seed: int = 42) -> Generator:
    """
    Initialize (or reinitialize) the global random Generator.

    Must be called once at the start of the pipeline, before any
    generator or engine is invoked.

    Parameters
    ----------
    seed:
        Integer seed for reproducibility. Defaults to 42.

    Returns
    -------
    numpy.random.Generator
        The initialized Generator instance.
    """
    global _rng, _seed
    _seed = seed
    _rng = np.random.default_rng(seed=seed)
    return _rng


def get_rng() -> Generator:
    """
    Return the shared random Generator instance.

    Raises
    ------
    RuntimeError
        If ``init_rng`` has not been called yet.

    Returns
    -------
    numpy.random.Generator
    """
    if _rng is None:
        raise RuntimeError(
            "Random Generator has not been initialized. "
            "Call `init_rng(seed=<int>)` before using `get_rng()`."
        )
    return _rng


def get_seed() -> int | None:
    """Return the seed that was used to initialize the Generator."""
    return _seed


def fork_rng(child_seed_offset: int = 0) -> Generator:
    """
    Create an independent child Generator derived from the global seed.

    Useful when a sub-component needs its own isolated Generator
    without affecting the global state (e.g., in tests).

    Parameters
    ----------
    child_seed_offset:
        An integer added to the global seed to create a distinct child.

    Returns
    -------
    numpy.random.Generator
    """
    if _seed is None:
        raise RuntimeError("Global seed not initialized. Call `init_rng()` first.")
    return np.random.default_rng(seed=_seed + child_seed_offset)
