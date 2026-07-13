"""
Base response schemas for API endpoints.

Provides standardized response formats for success and error cases.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response model with success indicator."""

    success: bool = Field(
        ...,
        description="Whether the request was successful"
    )

    message: str = Field(
        ...,
        description="Human-readable message describing the result"
    )


class SuccessResponse(BaseResponse):
    """Success response with optional data payload."""

    success: bool = Field(default=True, frozen=True)

    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Response data payload"
    )


class ErrorResponse(BaseResponse):
    """Error response with optional detail."""

    success: bool = Field(default=False, frozen=True)

    detail: Optional[Any] = Field(
        default=None,
        description="Additional error details (validation errors, stack trace, etc)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "message": "Validation error",
                "detail": [
                    {
                        "loc": ["body", "age_month"],
                        "msg": "Input should be less than or equal to 60",
                        "type": "less_than_equal"
                    }
                ]
            }
        }
    }
