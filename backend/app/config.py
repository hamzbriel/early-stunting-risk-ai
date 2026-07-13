"""
Configuration management for the backend application.

Uses pathlib for cross-platform path handling and Pydantic Settings
for type-safe configuration with environment variable support.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application metadata
    app_name: str = "Early Stunting Risk AI"
    app_version: str = "1.0.0"
    app_description: str = "End-to-End AI System for Early Stunting Risk Prediction"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # CORS settings
    cors_origins: list[str] = ["*"]

    # Logging
    log_level: str = "INFO"

    # Path configuration (computed properties)
    @property
    def PROJECT_ROOT(self) -> Path:
        """Root directory of the entire project (early-stunting-risk-ai/)."""
        # Backend is at: PROJECT_ROOT/backend/app/config.py
        # So we go up 2 levels: app -> backend -> PROJECT_ROOT
        return Path(__file__).resolve().parent.parent.parent

    @property
    def BACKEND_ROOT(self) -> Path:
        """Backend directory root."""
        return self.PROJECT_ROOT / "backend"

    @property
    def APP_ROOT(self) -> Path:
        """Application root (backend/app/)."""
        return self.BACKEND_ROOT / "app"

    # Model artifacts paths
    @property
    def MODEL_DIR(self) -> Path:
        """Directory containing all model artifacts."""
        return self.PROJECT_ROOT / "model"

    @property
    def TRAINED_MODEL_DIR(self) -> Path:
        """Directory containing trained model files."""
        return self.MODEL_DIR / "trained_models"

    @property
    def ARTIFACT_DIR(self) -> Path:
        """Directory containing model artifacts (encoders, configs, etc)."""
        return self.MODEL_DIR / "artifacts"

    @property
    def EXPLAINABILITY_DIR(self) -> Path:
        """Directory containing explainability artifacts (SHAP results)."""
        return self.MODEL_DIR / "explainability"

    # Specific artifact file paths
    @property
    def MODEL_PATH(self) -> Path:
        """Path to the trained model pipeline."""
        return self.TRAINED_MODEL_DIR / "best_pipeline.pkl"

    @property
    def LABEL_ENCODER_PATH(self) -> Path:
        """Path to the label encoder."""
        return self.ARTIFACT_DIR / "label_encoder.pkl"

    @property
    def FEATURE_NAMES_PATH(self) -> Path:
        """Path to feature names JSON."""
        return self.ARTIFACT_DIR / "feature_names.json"

    @property
    def TRAINING_CONFIG_PATH(self) -> Path:
        """Path to training configuration JSON."""
        return self.ARTIFACT_DIR / "training_config.json"

    @property
    def EVALUATION_RESULTS_PATH(self) -> Path:
        """Path to evaluation results JSON."""
        return self.ARTIFACT_DIR / "evaluation_results.json"

    @property
    def FEATURE_IMPORTANCE_PATH(self) -> Path:
        """Path to feature importance CSV."""
        return self.EXPLAINABILITY_DIR / "feature_importance.csv"

    @property
    def EXPLANATION_SUMMARY_PATH(self) -> Path:
        """Path to explanation summary JSON."""
        return self.EXPLAINABILITY_DIR / "explanation_summary.json"

    # Static files and templates
    @property
    def STATIC_DIR(self) -> Path:
        """Directory for static files (CSS, JS, images)."""
        return self.APP_ROOT / "static"

    @property
    def TEMPLATES_DIR(self) -> Path:
        """Directory for Jinja2 templates."""
        return self.APP_ROOT / "templates"

    def validate_paths(self) -> dict[str, bool]:
        """
        Validate that all required model artifacts exist.

        Returns:
            Dictionary mapping path names to existence status
        """
        required_paths = {
            "MODEL_PATH": self.MODEL_PATH,
            "LABEL_ENCODER_PATH": self.LABEL_ENCODER_PATH,
            "FEATURE_NAMES_PATH": self.FEATURE_NAMES_PATH,
            "TRAINING_CONFIG_PATH": self.TRAINING_CONFIG_PATH,
            "EVALUATION_RESULTS_PATH": self.EVALUATION_RESULTS_PATH,
            "FEATURE_IMPORTANCE_PATH": self.FEATURE_IMPORTANCE_PATH,
            "EXPLANATION_SUMMARY_PATH": self.EXPLANATION_SUMMARY_PATH,
        }

        return {name: path.exists() for name, path in required_paths.items()}

    def get_missing_paths(self) -> list[str]:
        """
        Get list of missing required artifacts.

        Returns:
            List of path names that don't exist
        """
        validation = self.validate_paths()
        return [name for name, exists in validation.items() if not exists]


# Global settings instance (singleton)
settings = Settings()


# Convenience function for testing/debugging
def print_config() -> None:
    """Print all configuration paths for debugging."""
    print(f"\n{'='*60}")
    print(f"Configuration for {settings.app_name} v{settings.app_version}")
    print(f"{'='*60}\n")

    print("Directory Paths:")
    print(f"  PROJECT_ROOT:        {settings.PROJECT_ROOT}")
    print(f"  BACKEND_ROOT:        {settings.BACKEND_ROOT}")
    print(f"  MODEL_DIR:           {settings.MODEL_DIR}")
    print(f"  TRAINED_MODEL_DIR:   {settings.TRAINED_MODEL_DIR}")
    print(f"  ARTIFACT_DIR:        {settings.ARTIFACT_DIR}")
    print(f"  EXPLAINABILITY_DIR:  {settings.EXPLAINABILITY_DIR}")
    print(f"  STATIC_DIR:          {settings.STATIC_DIR}")
    print(f"  TEMPLATES_DIR:       {settings.TEMPLATES_DIR}")

    print("\nArtifact Files:")
    print(f"  MODEL:               {settings.MODEL_PATH}")
    print(f"  LABEL_ENCODER:       {settings.LABEL_ENCODER_PATH}")
    print(f"  FEATURE_NAMES:       {settings.FEATURE_NAMES_PATH}")
    print(f"  TRAINING_CONFIG:     {settings.TRAINING_CONFIG_PATH}")
    print(f"  EVALUATION_RESULTS:  {settings.EVALUATION_RESULTS_PATH}")
    print(f"  FEATURE_IMPORTANCE:  {settings.FEATURE_IMPORTANCE_PATH}")
    print(f"  EXPLANATION_SUMMARY: {settings.EXPLANATION_SUMMARY_PATH}")

    print("\nPath Validation:")
    validation = settings.validate_paths()
    for name, exists in validation.items():
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}")

    missing = settings.get_missing_paths()
    if missing:
        print(f"\nWARNING: Missing {len(missing)} required artifacts:")
        for name in missing:
            print(f"  - {name}")
    else:
        print("\nAll required artifacts found!")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print_config()
