"""
Pydantic schemas package.

Exports all request, response, prediction, and model info schemas.
"""

from app.schemas.model_info import (
    FeatureImportance,
    ModelInfoResponse,
    ModelMetrics,
)
from app.schemas.prediction import PredictionResponse, RiskFactor
from app.schemas.request import PredictionRequest
from app.schemas.response import BaseResponse, ErrorResponse, SuccessResponse

__all__ = [
    # Request schemas
    "PredictionRequest",
    # Response schemas
    "BaseResponse",
    "SuccessResponse",
    "ErrorResponse",
    # Prediction schemas
    "PredictionResponse",
    "RiskFactor",
    # Model info schemas
    "ModelInfoResponse",
    "ModelMetrics",
    "FeatureImportance",
]
