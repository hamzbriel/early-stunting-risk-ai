"""
core/pipeline.py - Main orchestration pipeline.

Coordinates all pipeline stages in the correct order:
    1. Initialize RNG
    2. Generate base features (child, mother, father, household, nutrition, healthcare)
    3. Apply relationship engine (adjust features based on dependencies)
    4. Compute risk score and label
    5. Inject configurable noise
    6. Validate the dataset
    7. Split into train / validation / test
    8. Export all artifacts
    9. Generate HTML reports

Each stage receives the DatasetBuilder (shared state) and the GeneratorConfig.
Stages are isolated - they only depend on what has been added to the builder.
"""

import time

import pandas as pd

from synthetic_data.src.core.config_loader import GeneratorConfig, load_config
from synthetic_data.src.core.dataset_builder import DatasetBuilder
from synthetic_data.src.core.random_manager import RandomManager
from synthetic_data.src.engines.noise_engine import NoiseEngine
from synthetic_data.src.engines.relationship_engine import RelationshipEngine
from synthetic_data.src.engines.risk_engine import RiskEngine
from synthetic_data.src.exporters.csv_exporter import CSVExporter
from synthetic_data.src.exporters.dictionary_exporter import DictionaryExporter
from synthetic_data.src.exporters.metadata_exporter import MetadataExporter
from synthetic_data.src.exporters.statistics_exporter import StatisticsExporter
from synthetic_data.src.generators.child import ChildGenerator
from synthetic_data.src.generators.father import FatherGenerator
from synthetic_data.src.generators.healthcare import HealthcareGenerator
from synthetic_data.src.generators.household import HouseholdGenerator
from synthetic_data.src.generators.mother import MotherGenerator
from synthetic_data.src.generators.nutrition import NutritionGenerator
from synthetic_data.src.reports.quality_report import QualityReportGenerator
from synthetic_data.src.utils.constants import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION
from synthetic_data.src.utils.logger import get_logger, set_global_log_level
from synthetic_data.src.validators.dependency_validator import DependencyValidator
from synthetic_data.src.validators.distribution_validator import DistributionValidator
from synthetic_data.src.validators.duplicate_validator import DuplicateValidator
from synthetic_data.src.validators.range_validator import RangeValidator

logger = get_logger(__name__)


class SyntheticDataPipeline:
    """
    Orchestrates the entire synthetic data generation process.

    Parameters
    ----------
    config:
        Fully loaded and validated GeneratorConfig.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.n_samples: int = config.generator.n_samples

        # Shared state
        self._builder: DatasetBuilder = DatasetBuilder(n_samples=self.n_samples)
        self._random_manager: RandomManager = RandomManager(config)

        # Pipeline outputs
        self.datasets: dict[str, pd.DataFrame] = {}
        self.validation_report: dict = {}

    # Public API

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Execute the full pipeline end-to-end.

        Returns
        -------
        dict[str, pd.DataFrame]
            A mapping of split name → DataFrame.
            Keys: 'train', 'validation', 'test'.
        """
        pipeline_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("Synthetic Data Pipeline - START")
        logger.info("n_samples=%d | seed=%d", self.n_samples, self.config.generator.seed)
        logger.info("=" * 60)

        # Stage 0: Initialize RNG
        self._stage_init_rng()

        # Stage 1: Feature Generation
        self._stage_generate_features()

        # Stage 2: Relationship Engine
        self._stage_apply_relationships()

        # Stage 3: Risk Engine
        self._stage_compute_risk()

        # Stage 4: Noise Engine
        self._stage_inject_noise()

        # Stage 5: Validation
        self._stage_validate()

        # Stage 6: Split
        self._stage_split()

        # Stage 7: Export
        self._stage_export()

        # Stage 8: Reports
        self._stage_reports()

        elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 60)
        logger.info("Pipeline completed in %.2f seconds.", elapsed)
        logger.info("=" * 60)

        return self.datasets

    # Private - Pipeline Stages

    def _stage_init_rng(self) -> None:
        logger.info("[Stage 0] Initializing Random Number Generator...")
        self._random_manager.initialize()

    def _stage_generate_features(self) -> None:
        logger.info("[Stage 1] Generating base features...")
        rng = self._random_manager.rng
        cfg = self.config

        generators = [
            ChildGenerator(cfg, rng),
            MotherGenerator(cfg, rng),
            FatherGenerator(cfg, rng),
            HouseholdGenerator(cfg, rng),
            NutritionGenerator(cfg, rng),
            HealthcareGenerator(cfg, rng),
        ]

        for gen in generators:
            logger.info("  -> %s", gen.__class__.__name__)
            columns = gen.generate(self.n_samples)
            self._builder.add_columns(columns)

        logger.info(
            "  Features generated: %d columns", len(self._builder.column_names)
        )

    def _stage_apply_relationships(self) -> None:
        logger.info("[Stage 2] Applying Relationship Engine (DAG)...")
        engine = RelationshipEngine(self.config, self._random_manager.rng)
        engine.apply(self._builder)

    def _stage_compute_risk(self) -> None:
        logger.info("[Stage 3] Computing risk scores and labels...")
        engine = RiskEngine(self.config, self._random_manager.rng)
        engine.apply(self._builder)

    def _stage_inject_noise(self) -> None:
        logger.info("[Stage 4] Injecting noise...")
        engine = NoiseEngine(self.config, self._random_manager.rng)
        engine.apply(self._builder)

    def _stage_validate(self) -> None:
        logger.info("[Stage 5] Validating dataset...")
        df = self._builder.to_dataframe()

        validators = [
            RangeValidator(self.config),
            DuplicateValidator(self.config),
            DependencyValidator(self.config),
            DistributionValidator(self.config),
        ]

        all_issues: list[dict] = []
        for validator in validators:
            issues = validator.validate(df)
            all_issues.extend(issues)
            if issues:
                for issue in issues:
                    severity = issue.get("severity", "warning").upper()
                    logger.warning("  [%s] %s: %s", severity, issue["rule"], issue["message"])
            else:
                logger.info("  [OK] %s - no issues", validator.__class__.__name__)

        self.validation_report = {
            "total_issues": len(all_issues),
            "issues": all_issues,
        }
        logger.info("  Validation complete: %d issue(s) found.", len(all_issues))

    def _stage_split(self) -> None:
        logger.info("[Stage 6] Splitting dataset...")
        from sklearn.model_selection import train_test_split

        df = self._builder.to_dataframe()
        split_cfg = self.config.generator.split
        stratify_col = self.config.generator.stratify_column

        stratify = df[stratify_col] if stratify_col in df.columns else None

        # First split: train vs (val + test)
        val_test_ratio = split_cfg.validation + split_cfg.test
        df_train, df_val_test = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=self.config.generator.seed,
            stratify=stratify,
        )

        # Second split: val vs test
        stratify_val = df_val_test[stratify_col] if stratify_col in df_val_test.columns else None
        test_ratio_within = split_cfg.test / val_test_ratio

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=test_ratio_within,
            random_state=self.config.generator.seed,
            stratify=stratify_val,
        )

        # Reset indices
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        self.datasets = {
            SPLIT_TRAIN: df_train,
            SPLIT_VALIDATION: df_val,
            SPLIT_TEST: df_test,
        }

        logger.info(
            "  Split: train=%d | validation=%d | test=%d",
            len(df_train), len(df_val), len(df_test),
        )

    def _stage_export(self) -> None:
        logger.info("[Stage 7] Exporting artifacts...")
        full_df = self._builder.to_dataframe()

        CSVExporter(self.config).export(self.datasets)
        MetadataExporter(self.config).export(self.datasets, self.validation_report)
        StatisticsExporter(self.config).export(full_df)
        DictionaryExporter(self.config).export(full_df)

    def _stage_reports(self) -> None:
        logger.info("[Stage 8] Generating HTML reports...")
        full_df = self._builder.to_dataframe()
        QualityReportGenerator(self.config).generate(full_df, self.validation_report)


# Convenience Factory


def run_pipeline(config_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Convenience function to run the full pipeline with one call.

    Parameters
    ----------
    config_dir:
        Optional path to config directory.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    set_global_log_level("INFO")
    config = load_config(config_dir=config_dir)
    pipeline = SyntheticDataPipeline(config)
    return pipeline.run()
