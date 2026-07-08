"""
core/config_loader.py - Configuration loader with Pydantic validation.

Loads all YAML configuration files and validates them using Pydantic models.
This ensures that any configuration error is caught at startup, not
mid-pipeline.

Architecture note:
    GeneratorConfig is the single source of truth for all pipeline
    configuration. All modules receive it as a dependency - they do NOT
    read YAML files directly.

Usage:
    from synthetic_data.src.core.config_loader import load_config

    config = load_config(config_dir="synthetic_data/config")
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from synthetic_data.src.utils.logger import get_logger
from synthetic_data.src.utils.yaml_reader import load_yaml

logger = get_logger(__name__)


# Pydantic Models

class SplitConfig(BaseModel):
    """Train / validation / test split ratios."""

    train: float = Field(0.70, ge=0.0, le=1.0)
    validation: float = Field(0.15, ge=0.0, le=1.0)
    test: float = Field(0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sum(self) -> "SplitConfig":
        total = round(self.train + self.validation + self.test, 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class GeneratorMainConfig(BaseModel):
    """Top-level generator settings from generator.yaml."""

    n_samples: int = Field(10_000, ge=100, le=1_000_000)
    seed: int = Field(42)
    version: str = "1.0.0"
    description: str = ""
    split: SplitConfig = Field(default_factory=SplitConfig)
    stratify_column: str = "risk_level"
    output_dir: str = "output"
    reports_dir: str = "reports"
    log_level: str = "INFO"
    debug_mode: bool = False

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


class GeneratorConfig(BaseModel):
    """
    Aggregate configuration object passed through the entire pipeline.

    Holds all sub-configs loaded from separate YAML files.
    """

    # Main generator settings
    generator: GeneratorMainConfig

    # Raw dicts from sub-config files (validated structurally by Pydantic below)
    distributions: dict[str, Any]
    relationships: dict[str, Any]
    risk_rules: dict[str, Any]
    validation_rules: dict[str, Any]
    export: dict[str, Any]

    # Resolved paths (set after loading)
    config_dir: Path
    output_dir: Path
    reports_dir: Path

    model_config = {"arbitrary_types_allowed": True}


# Loader Function


def load_config(config_dir: str | Path | None = None) -> GeneratorConfig:
    """
    Load and validate all configuration YAML files.

    Parameters
    ----------
    config_dir:
        Path to the config/ directory. Defaults to the standard location
        relative to this file: ``synthetic_data/config/``.

    Returns
    -------
    GeneratorConfig
        A fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If any required config file is missing.
    pydantic.ValidationError
        If any config value fails validation.
    """
    if config_dir is None:
        # Default: two levels up from core/ → synthetic_data/config/
        config_dir = Path(__file__).resolve().parents[2] / "config"

    config_dir = Path(config_dir).resolve()
    logger.info("Loading configuration from: %s", config_dir)

    # Load main generator config
    generator_raw = load_yaml(config_dir / "generator.yaml")
    generator_cfg = GeneratorMainConfig(**generator_raw.get("generator", generator_raw))

    # Resolve output paths
    synthetic_root = config_dir.parent
    output_dir = synthetic_root / generator_cfg.output_dir
    reports_dir = synthetic_root / generator_cfg.reports_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load sub-configs
    distributions = load_yaml(config_dir / "distributions.yaml")
    relationships = load_yaml(config_dir / "relationships.yaml")
    risk_rules = load_yaml(config_dir / "risk_rules.yaml")
    validation_rules = load_yaml(config_dir / "validation.yaml")
    export_cfg = load_yaml(config_dir / "export.yaml")

    config = GeneratorConfig(
        generator=generator_cfg,
        distributions=distributions,
        relationships=relationships,
        risk_rules=risk_rules,
        validation_rules=validation_rules,
        export=export_cfg,
        config_dir=config_dir,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )

    logger.info(
        "Configuration loaded: n_samples=%d, seed=%d, version=%s",
        config.generator.n_samples,
        config.generator.seed,
        config.generator.version,
    )
    return config
