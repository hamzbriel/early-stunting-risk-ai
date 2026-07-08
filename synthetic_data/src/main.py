"""
main.py — Main execution entry point for the Synthetic Data Platform.

Loads configuration, runs the orchestrator pipeline, and generates
the final synthetic datasets and validation audit reports.

Usage:
    # Run from root of repository
    python -m synthetic_data.src.main

    # Or run from synthetic_data directory
    python src/main.py
"""

import sys
from pathlib import Path

# Add current folder to path to allow direct script executions if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))

from synthetic_data.src.core.pipeline import run_pipeline
from synthetic_data.src.utils.logger import get_logger

logger = get_logger("synthetic_data.main")


def main() -> None:
    """
    Entry point to run the Synthetic Data Platform generation.
    """
    try:
        # Run with default config folder location (inside synthetic_data/config)
        run_pipeline()
        logger.info("Generation process completed successfully!")
    except Exception as e:
        logger.exception("An error occurred during data generation: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
