"""
Model information schemas for model metadata and statistics endpoints.

Provides schemas for model info, feature importance, and training metrics.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    """Feature importance entry."""

    feature: str = Field(
        ...,
        description="Feature name"
    )

    importance: float = Field(
        ...,
        description="Importance score"
    )


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""

    accuracy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model accuracy"
    )

    precision: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model precision"
    )

    recall: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model recall"
    )

    f1_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model F1 score"
    )


class ModelInfoResponse(BaseModel):
    """Comprehensive model information response."""

    model_type: str = Field(
        ...,
        description="Type of the trained model"
    )

    num_features: int = Field(
        ...,
        ge=0,
        description="Number of input features"
    )

    num_classes: int = Field(
        ...,
        ge=2,
        description="Number of target classes"
    )

    classes: list[str] = Field(
        ...,
        description="List of class labels"
    )

    training_date: Optional[str] = Field(
        default=None,
        description="Date when model was trained"
    )

    model_name: Optional[str] = Field(
        default=None,
        description="Model algorithm name"
    )

    metrics: Optional[ModelMetrics] = Field(
        default=None,
        description="Model evaluation metrics"
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "model_type": "Pipeline",
                "num_features": 20,
                "num_classes": 3,
                "classes": ["Low", "Medium", "High"],
                "training_date": "2024-01-15",
                "model_name": "XGBoost",
                "metrics": {
                    "accuracy": 0.92,
                    "precision": 0.91,
                    "recall": 0.90,
                    "f1_score": 0.90
                }
            }
        }
    }
