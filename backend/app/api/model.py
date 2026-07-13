"""
Model information endpoints.

Provides metadata about the loaded ML model including
configuration, metrics, and feature information.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.model_loader import get_model_loader
from app.schemas.model_info import ModelInfoResponse

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """
    Get comprehensive model information.

    Returns metadata about the loaded model including:
    - Model type and algorithm
    - Number of features and classes
    - Class labels
    - Training configuration
    - Evaluation metrics

    Returns:
        ModelInfoResponse with complete model metadata
    """
    try:
        loader = get_model_loader()
        info = loader.get_model_info()

        return ModelInfoResponse(**info)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model information: {str(e)}"
        ) from e
