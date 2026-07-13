"""
Prediction endpoints.

Handles stunting risk prediction requests and returns
structured responses with risk levels, probabilities,
and personalized recommendations.
"""

from fastapi import APIRouter, HTTPException

from app.core.predictor import get_predictor
from app.schemas.prediction import PredictionResponse
from app.schemas.request import PredictionRequest

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict stunting risk level for a child.

    Accepts child health and demographic information and returns
    a risk assessment including:
    - Predicted risk level (Low/Medium/High)
    - Confidence score (0-100%)
    - Probability distribution across all risk levels
    - Top contributing risk factors
    - Personalized health recommendations

    Args:
        request: Validated prediction request with all required features

    Returns:
        PredictionResponse with complete risk assessment

    Raises:
        422: Validation error - invalid input values
        500: Prediction failed - model inference error
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(request)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input: {str(e)}"
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during prediction: {str(e)}"
        ) from e
