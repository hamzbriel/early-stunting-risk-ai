"""
Model Loader Service - Singleton pattern for loading ML artifacts.

This module handles loading all machine learning artifacts (model, encoders,
configurations) once at application startup and provides cached access
throughout the application lifetime.
"""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from app.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Singleton class for loading and caching ML model artifacts.

    All artifacts are loaded once during initialization and cached
    for the lifetime of the application. Subsequent calls to get_instance()
    return the same cached instance.
    """

    _instance: Optional["ModelLoader"] = None

    def __init__(self) -> None:
        """
        Initialize the model loader and load all artifacts.

        This should not be called directly. Use get_instance() instead.
        """
        if ModelLoader._instance is not None:
            raise RuntimeError(
                "ModelLoader is a singleton. Use ModelLoader.get_instance() instead."
            )

        logger.info("Initializing ModelLoader...")

        # Required artifacts
        self.model = self._load_model()
        self.label_encoder = self._load_label_encoder()
        self.feature_names = self._load_feature_names()

        # Optional artifacts (won't crash if missing)
        self.training_config = self._load_training_config()
        self.evaluation_results = None
        self.explanation_summary = self._load_explanation_summary()
        self.feature_importance = self._load_feature_importance()

        logger.info("ModelLoader initialized successfully")

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """
        Get the singleton instance of ModelLoader.

        Returns:
            Cached ModelLoader instance

        Example:
            loader = ModelLoader.get_instance()
            model = loader.model
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (useful for testing).

        This forces a reload of all artifacts on next get_instance() call.
        """
        cls._instance = None

    def _load_model(self) -> Any:
        """Load the trained model pipeline."""
        path = settings.MODEL_PATH
        logger.info(f"Loading model from: {path.name}")

        if not path.exists():
            logger.error(f"Model file not found: {path}")
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Please ensure the ML pipeline has been trained."
            )

        try:
            model = joblib.load(path)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_label_encoder(self) -> Any:
        """Load the label encoder."""
        path = settings.LABEL_ENCODER_PATH
        logger.info(f"Loading label encoder from: {path.name}")

        if not path.exists():
            logger.error(f"Label encoder not found: {path}")
            raise FileNotFoundError(
                f"Label encoder not found: {path}\n"
                "Please ensure the ML pipeline has been trained."
            )

        try:
            encoder = joblib.load(path)
            logger.info(f"Label encoder loaded: {encoder.classes_.tolist()}")
            return encoder
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
            raise RuntimeError(f"Failed to load label encoder: {e}")

    def _load_feature_names(self) -> list[str]:
        """Load feature names from JSON."""
        path = settings.FEATURE_NAMES_PATH
        logger.info(f"Loading feature names from: {path.name}")

        if not path.exists():
            logger.error(f"Feature names file not found: {path}")
            raise FileNotFoundError(
                f"Feature names file not found: {path}\n"
                "Please ensure the ML pipeline has been trained."
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both formats: {"features": [...]} or [...]
            features = data if isinstance(data, list) else data.get("features", [])

            if not features:
                raise ValueError("Feature names list is empty")

            logger.info(f"Loaded {len(features)} features")
            return features
        except Exception as e:
            logger.error(f"Failed to load feature names: {e}")
            raise RuntimeError(f"Failed to load feature names: {e}")

    def _load_training_config(self) -> Optional[dict[str, Any]]:
        """Load training configuration (optional)."""
        path = settings.TRAINING_CONFIG_PATH
        logger.info(f"Loading training config from: {path.name}")

        if not path.exists():
            logger.warning("Training config not found (optional)")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("Training config loaded successfully")
            return config
        except Exception as e:
            logger.warning(f"Failed to load training config: {e}")
            return None

    def _load_explanation_summary(self) -> Optional[dict[str, Any]]:
        """Load SHAP explanation summary (optional)."""
        path = settings.EXPLANATION_SUMMARY_PATH
        logger.info(f"Loading explanation summary from: {path.name}")

        if not path.exists():
            logger.warning("Explanation summary not found (optional)")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            logger.info("Explanation summary loaded successfully")
            return summary
        except Exception as e:
            logger.warning(f"Failed to load explanation summary: {e}")
            return None

    def _load_feature_importance(self) -> Optional[pd.DataFrame]:
        """Load feature importance from CSV (optional)."""
        path = settings.FEATURE_IMPORTANCE_PATH
        logger.info(f"Loading feature importance from: {path.name}")

        if not path.exists():
            logger.warning("Feature importance not found (optional)")
            return None

        try:
            df = pd.read_csv(path)
            logger.info(f"Feature importance loaded: {len(df)} features")
            return df
        except Exception as e:
            logger.warning(f"Failed to load feature importance: {e}")
            return None

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model metadata, configuration, and metrics
        """
        info: dict[str, Any] = {
            "model_type": type(self.model).__name__,
            "num_features": len(self.feature_names),
            "num_classes": len(self.label_encoder.classes_),
            "classes": self.label_encoder.classes_.tolist(),
        }

        # Add training config if available
        if self.training_config:
            info["training_date"] = self.training_config.get("training_date")
            info["model_name"] = self.training_config.get("model_name")

        # Add evaluation metrics if available
        if self.evaluation_results:
            info["metrics"] = self.evaluation_results.get("test_metrics", {})

        return info


def get_model_loader() -> ModelLoader:
    """
    Convenience function to get the ModelLoader singleton instance.

    Returns:
        ModelLoader instance

    Example:
        loader = get_model_loader()
        predictions = loader.model.predict(X)
    """
    return ModelLoader.get_instance()
