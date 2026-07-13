"""
Prediction Service - Handles ML inference for stunting risk prediction.

This service provides a clean interface between the API layer and the
ML model, handling data transformation, prediction, and result formatting.
"""

from typing import Any

import pandas as pd

from app.core.logger import get_logger
from app.core.model_loader import get_model_loader
from app.schemas.prediction import PredictionResponse, RiskFactor
from app.schemas.request import PredictionRequest

logger = get_logger(__name__)


class Predictor:
    """
    Prediction service for stunting risk assessment.

    Uses the loaded model pipeline to make predictions and formats
    the results into structured responses.
    """

    def __init__(self) -> None:
        """Initialize predictor with loaded model artifacts."""
        self.loader = get_model_loader()
        self.model = self.loader.model
        self.label_encoder = self.loader.label_encoder
        self.feature_names = self.loader.feature_names
        self.feature_importance_df = self.loader.feature_importance

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a prediction for the given input.

        Args:
            request: Validated prediction request with all features

        Returns:
            Structured prediction response with risk level, confidence,
            probabilities, and top risk factors

        Raises:
            ValueError: If prediction fails due to invalid input
            RuntimeError: If model inference fails
        """
        logger.info("Starting prediction request")

        # Convert Pydantic model to dict, then to DataFrame
        input_dict = request.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Ensure column order matches training (model pipeline handles this, but be explicit)
        # The pipeline's ColumnTransformer will handle preprocessing

        try:
            # Get class probabilities
            probabilities = self.model.predict_proba(input_df)[0]

            # Get predicted class index
            prediction_idx = probabilities.argmax()

            # Decode prediction label
            prediction_label = self.label_encoder.inverse_transform([prediction_idx])[0]

            # Get confidence (probability of predicted class)
            confidence = float(probabilities[prediction_idx] * 100)

            # Build probability distribution dict
            prob_dict = {
                label: float(prob * 100)
                for label, prob in zip(self.label_encoder.classes_, probabilities)
            }

            # Get top risk factors
            top_factors = self._get_top_risk_factors(input_dict, n=5)

            # Generate recommendation
            recommendation = self._generate_recommendation(prediction_label, input_dict)

            logger.info(
                f"Prediction completed: {prediction_label} "
                f"(confidence: {confidence:.1f}%)"
            )

            return PredictionResponse(
                prediction=prediction_label,
                confidence=confidence,
                probabilities=prob_dict,
                top_risk_factors=top_factors,
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def _get_top_risk_factors(
        self, input_dict: dict[str, Any], n: int = 5
    ) -> list[RiskFactor]:
        """
        Get top N risk factors based on feature importance.

        Uses global feature importance from SHAP analysis.
        For more accurate local explanations, SHAP would need to be
        computed per-prediction (expensive).

        Args:
            input_dict: Input feature values
            n: Number of top factors to return

        Returns:
            List of top risk factors sorted by importance
        """
        if self.feature_importance_df is None:
            return []

        try:
            # Filter to features that are in the input
            # Feature importance CSV has original feature names (before encoding)
            importance_data = self.feature_importance_df.copy()

            # Sort by importance and take top N
            top_n = importance_data.nlargest(n, "importance")

            return [
                RiskFactor(
                    feature=row["feature"],
                    importance=float(row["importance"]),
                )
                for _, row in top_n.iterrows()
            ]

        except Exception as e:
            print(f"Warning: Failed to get risk factors: {e}")
            return []

    def _generate_recommendation(
        self, prediction: str, input_dict: dict[str, Any]
    ) -> str:
        """
        Generate health recommendation based on prediction and input features.

        Args:
            prediction: Predicted risk level
            input_dict: Input feature values

        Returns:
            Personalized health recommendation
        """
        recommendations = {
            "Low": (
                "Continue current health practices. "
                "Maintain regular growth monitoring and balanced nutrition."
            ),
            "Medium": (
                "Monitor child's growth regularly and ensure adequate protein intake. "
                "Consider consulting healthcare provider for nutritional guidance."
            ),
            "High": (
                "Immediate attention recommended. "
                "Consult healthcare provider for comprehensive nutritional assessment "
                "and intervention plan."
            ),
        }

        base_recommendation = recommendations.get(
            prediction,
            "Consult healthcare provider for personalized guidance.",
        )

        # Add specific recommendations based on input
        specific_advice = []

        if input_dict.get("protein_intake") == "low":
            specific_advice.append("Increase protein-rich foods in child's diet.")

        if input_dict.get("vitamin_intake") == "low":
            specific_advice.append("Ensure adequate vitamin supplementation.")

        if input_dict.get("exclusive_breastfeeding") == 0:
            if input_dict.get("age_month", 0) < 6:
                specific_advice.append(
                    "Consider exclusive breastfeeding for first 6 months if possible."
                )

        if input_dict.get("immunization") in ["none", "partial"]:
            specific_advice.append("Complete immunization schedule.")

        if input_dict.get("healthcare_access") == "poor":
            specific_advice.append(
                "Seek improved access to healthcare services for regular monitoring."
            )

        if specific_advice:
            return f"{base_recommendation} {' '.join(specific_advice)}"

        return base_recommendation

    def get_feature_info(self) -> dict[str, Any]:
        """
        Get information about expected features.

        Returns:
            Dictionary with feature names and count
        """
        return {
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }


def get_predictor() -> Predictor:
    """
    Get a Predictor instance.

    Returns:
        Predictor instance with loaded model

    Example:
        predictor = get_predictor()
        result = predictor.predict(request)
    """
    return Predictor()
