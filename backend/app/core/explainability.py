"""
Explainability Service - Provides access to pre-computed SHAP results.

This service does NOT compute SHAP values (too expensive for runtime).
Instead, it provides read-only access to pre-computed feature importance
and explanation summaries from the training phase.
"""

from typing import Any, Optional

from app.core.model_loader import get_model_loader
from app.schemas.model_info import FeatureImportance


class ExplainabilityService:
    """
    Service for accessing pre-computed explainability artifacts.

    All SHAP computations are done during training phase (Phase 5).
    This service only reads and formats the results.
    """

    def __init__(self) -> None:
        """Initialize service with loaded explainability artifacts."""
        self.loader = get_model_loader()
        self.feature_importance_df = self.loader.feature_importance
        self.explanation_summary = self.loader.explanation_summary

    def get_feature_importance(self) -> list[FeatureImportance]:
        """
        Get global feature importance scores.

        Returns feature importance from pre-computed SHAP analysis.
        Sorted by importance score in descending order.

        Returns:
            List of feature importance entries

        Raises:
            ValueError: If feature importance data is not available
        """
        if self.feature_importance_df is None:
            raise ValueError(
                "Feature importance data not available. "
                "Ensure explainability artifacts were generated during training."
            )

        # Convert DataFrame to list of FeatureImportance objects
        result = [
            FeatureImportance(
                feature=str(row["feature"]),
                importance=float(row.get("importance", row.get("mean_abs_shap"))),
            )
            for _, row in self.feature_importance_df.iterrows()
        ]

        # Sort by importance (descending)
        result.sort(key=lambda x: x.importance, reverse=True)

        return result

    def get_explanation_summary(self) -> dict[str, Any]:
        """
        Get SHAP explanation summary.

        Returns pre-computed SHAP summary including global statistics
        and interpretation guidance.

        Returns:
            Explanation summary dictionary

        Raises:
            ValueError: If explanation summary is not available
        """
        if self.explanation_summary is None:
            raise ValueError(
                "Explanation summary not available. "
                "Ensure explainability artifacts were generated during training."
            )

        return self.explanation_summary

    def get_top_features(self, n: int = 10) -> list[FeatureImportance]:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of top N feature importance entries
        """
        all_features = self.get_feature_importance()
        return all_features[:n]

    def get_feature_importance_dict(self) -> dict[str, float]:
        """
        Get feature importance as a simple dict mapping.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        features = self.get_feature_importance()
        return {f.feature: f.importance for f in features}

    def is_available(self) -> bool:
        """
        Check if explainability data is available.

        Returns:
            True if both feature importance and explanation summary are loaded
        """
        return (
            self.feature_importance_df is not None
            and self.explanation_summary is not None
        )

    def get_availability_status(self) -> dict[str, bool]:
        """
        Get detailed availability status of explainability artifacts.

        Returns:
            Dictionary showing which artifacts are available
        """
        return {
            "feature_importance": self.feature_importance_df is not None,
            "explanation_summary": self.explanation_summary is not None,
            "fully_available": self.is_available(),
        }


def get_explainability_service() -> ExplainabilityService:
    """
    Get an ExplainabilityService instance.

    Returns:
        ExplainabilityService with loaded artifacts

    Example:
        service = get_explainability_service()
        features = service.get_feature_importance()
    """
    return ExplainabilityService()
