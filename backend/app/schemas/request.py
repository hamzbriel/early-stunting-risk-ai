"""
Request schemas for prediction endpoints.

Defines input validation for stunting risk prediction requests.
All numeric fields have range validation and categorical fields
have allowed value constraints.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Input schema for stunting risk prediction.

    All fields are validated for type, range, and allowed values.
    Categorical fields use Literal types for compile-time checking.
    """

    # Child Information
    age_month: int = Field(
        ...,
        ge=0,
        le=60,
        description="Child's age in months (0-60)"
    )

    gender: Literal["M", "F"] = Field(
        ...,
        description="Child's gender (M=Male, F=Female)"
    )

    birth_weight: float = Field(
        ...,
        gt=0,
        le=10.0,
        description="Birth weight in kg (must be positive, typically 1-6 kg)"
    )

    birth_length: float = Field(
        ...,
        gt=0,
        le=100.0,
        description="Birth length in cm (must be positive, typically 30-60 cm)"
    )

    # Mother Information
    mother_age: int = Field(
        ...,
        ge=15,
        le=60,
        description="Mother's age in years (15-60)"
    )

    mother_education: Literal["none", "primary", "secondary", "higher"] = Field(
        ...,
        description="Mother's education level"
    )

    mother_working: int = Field(
        ...,
        ge=0,
        le=1,
        description="Mother's working status (0=not working, 1=working)"
    )

    # Father Information
    father_education: Literal["none", "primary", "secondary", "higher"] = Field(
        ...,
        description="Father's education level"
    )

    father_working: int = Field(
        ...,
        ge=0,
        le=1,
        description="Father's working status (0=not working, 1=working)"
    )

    # Household Information
    family_income: Literal["very_low", "low", "medium", "high"] = Field(
        ...,
        description="Family income level"
    )

    sanitation: Literal["poor", "fair", "good"] = Field(
        ...,
        description="Household sanitation quality"
    )

    clean_water: int = Field(
        ...,
        ge=0,
        le=1,
        description="Access to clean water (0=no access, 1=has access)"
    )

    electricity: int = Field(
        ...,
        ge=0,
        le=1,
        description="Access to electricity (0=no access, 1=has access)"
    )

    house_density: Literal["low", "medium", "high"] = Field(
        ...,
        description="Household density/crowding level"
    )

    # Nutrition Information
    exclusive_breastfeeding: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exclusive breastfeeding for 6 months (0=no, 1=yes)"
    )

    protein_intake: Literal["low", "medium", "high"] = Field(
        ...,
        description="Child's protein intake level"
    )

    vitamin_intake: Literal["low", "medium", "high"] = Field(
        ...,
        description="Child's vitamin intake level"
    )

    # Healthcare Information
    immunization: Literal["none", "partial", "complete"] = Field(
        ...,
        description="Child's immunization status"
    )

    diarrhea_history: int = Field(
        ...,
        ge=0,
        le=1,
        description="History of diarrhea (0=no, 1=yes)"
    )

    healthcare_access: Literal["poor", "fair", "good"] = Field(
        ...,
        description="Access to healthcare services"
    )

    @field_validator("birth_weight")
    @classmethod
    def validate_birth_weight(cls, v: float) -> float:
        """Validate birth weight is in realistic range."""
        if v < 0.5 or v > 7.0:
            raise ValueError(
                "Birth weight should be between 0.5 and 7.0 kg. "
                "Please check the input value."
            )
        return v

    @field_validator("birth_length")
    @classmethod
    def validate_birth_length(cls, v: float) -> float:
        """Validate birth length is in realistic range."""
        if v < 25.0 or v > 65.0:
            raise ValueError(
                "Birth length should be between 25 and 65 cm. "
                "Please check the input value."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_month": 18,
                "gender": "M",
                "birth_weight": 3.2,
                "birth_length": 49.5,
                "mother_age": 28,
                "mother_education": "secondary",
                "mother_working": 1,
                "father_education": "secondary",
                "father_working": 1,
                "family_income": "medium",
                "sanitation": "good",
                "clean_water": 1,
                "electricity": 1,
                "house_density": "medium",
                "exclusive_breastfeeding": 1,
                "protein_intake": "medium",
                "vitamin_intake": "medium",
                "immunization": "complete",
                "diarrhea_history": 0,
                "healthcare_access": "good"
            }
        }
    }
