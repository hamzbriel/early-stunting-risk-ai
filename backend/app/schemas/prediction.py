"""
Prediction-specific schemas for stunting risk prediction responses.

Defines models for prediction results, risk factors, and probability distributions.
"""

from typing import Optional

from pydantic import BaseModel, Field


class RiskFactor(BaseModel):
    """Individual risk factor with importance score."""

    feature: str = Field(
        ...,
        description="Feature name"
    )

    importance: float = Field(
        ...,
        description="Feature importance score (0-1)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "feature": "protein_intake",
                "importance": 0.42
            }
        }
    }


class PredictionResponse(BaseModel):
    """Complete prediction response with risk assessment."""

    prediction: str = Field(
        ...,
        description="Predicted risk level (Low/Medium/High)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Confidence score (0-100%)"
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across all risk levels"
    )

    top_risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Top contributing risk factors"
    )

    recommendation: Optional[str] = Field(
        default=None,
        description="Health recommendation based on risk level"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Medium",
                "confidence": 98.4,
                "probabilities": {
                    "Low": 0.5,
                    "Medium": 98.4,
                    "High": 1.1
                },
                "top_risk_factors": [
                    {
                        "feature": "protein_intake",
                        "importance": 0.42
                    },
                    {
                        "feature": "birth_weight",
                        "importance": 0.35
                    },
                    {
                        "feature": "mother_education",
                        "importance": 0.28
                    }
                ],
                "recommendation": "Monitor child's growth regularly and ensure adequate protein intake."
            }
        }
    }
