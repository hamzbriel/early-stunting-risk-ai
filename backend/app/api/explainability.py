"""
Explainability endpoints.

Provides access to pre-computed SHAP analysis results including
feature importance scores and explanation summaries.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.explainability import get_explainability_service
from app.schemas.model_info import FeatureImportance

router = APIRouter()


@router.get("/feature-importance", response_model=list[FeatureImportance])
async def get_feature_importance() -> list[FeatureImportance]:
    """
    Get global feature importance scores.

    Returns feature importance from pre-computed SHAP analysis,
    sorted by importance in descending order.

    Returns:
        List of feature importance entries with scores
    """
    try:
        service = get_explainability_service()
        return service.get_feature_importance()

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feature importance: {str(e)}"
        ) from e


@router.get("/explanation-summary", response_model=dict[str, Any])
async def get_explanation_summary() -> dict[str, Any]:
    """
    Get SHAP explanation summary.

    Returns pre-computed SHAP summary including global statistics
    and interpretation guidance from training phase analysis.

    Returns:
        Explanation summary dictionary with SHAP insights
    """
    try:
        service = get_explainability_service()
        return service.get_explanation_summary()

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve explanation summary: {str(e)}"
        ) from e
