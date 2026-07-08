"""
utils/constants.py - Project-wide constants.

Centralizes string literals and magic numbers so they never appear
as hard-coded values scattered across the codebase.
"""

from pathlib import Path

# Directory Layout

# Root of the synthetic_data module
SYNTHETIC_DATA_ROOT: Path = Path(__file__).resolve().parents[3]

# Config directory
CONFIG_DIR: Path = SYNTHETIC_DATA_ROOT / "config"

# Output directory
OUTPUT_DIR: Path = SYNTHETIC_DATA_ROOT / "output"

# Reports directory
REPORTS_DIR: Path = SYNTHETIC_DATA_ROOT / "reports"

# Feature Names

CHILD_FEATURES: list[str] = [
    "age_month",
    "gender",
    "birth_weight",
    "birth_length",
]

MOTHER_FEATURES: list[str] = [
    "mother_age",
    "mother_education",
    "mother_working",
]

FATHER_FEATURES: list[str] = [
    "father_education",
    "father_working",
]

HOUSEHOLD_FEATURES: list[str] = [
    "family_income",
    "sanitation",
    "clean_water",
    "electricity",
    "house_density",
]

NUTRITION_FEATURES: list[str] = [
    "exclusive_breastfeeding",
    "protein_intake",
    "vitamin_intake",
]

HEALTHCARE_FEATURES: list[str] = [
    "immunization",
    "diarrhea_history",
    "healthcare_access",
]

TARGET_FEATURES: list[str] = [
    "risk_score",
    "risk_level",
]

ALL_FEATURES: list[str] = (
    CHILD_FEATURES
    + MOTHER_FEATURES
    + FATHER_FEATURES
    + HOUSEHOLD_FEATURES
    + NUTRITION_FEATURES
    + HEALTHCARE_FEATURES
    + TARGET_FEATURES
)

# Risk Levels

RISK_LEVELS: list[str] = ["Low", "Medium", "High"]

RISK_LEVEL_LOW: str = "Low"
RISK_LEVEL_MEDIUM: str = "Medium"
RISK_LEVEL_HIGH: str = "High"

# Feature Types (for data dictionary and validation)

NUMERIC_FEATURES: list[str] = [
    "age_month",
    "birth_weight",
    "birth_length",
    "mother_age",
    "risk_score",
]

BINARY_FEATURES: list[str] = [
    "mother_working",
    "father_working",
    "clean_water",
    "electricity",
    "exclusive_breastfeeding",
    "diarrhea_history",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "mother_education",
    "father_education",
    "family_income",
    "sanitation",
    "house_density",
    "protein_intake",
    "vitamin_intake",
    "immunization",
    "healthcare_access",
    "risk_level",
]

# Dataset Split Names

SPLIT_TRAIN: str = "train"
SPLIT_VALIDATION: str = "validation"
SPLIT_TEST: str = "test"

SPLIT_NAMES: list[str] = [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST]

# Default Config File Names

CONFIG_GENERATOR: str = "generator.yaml"
CONFIG_DISTRIBUTIONS: str = "distributions.yaml"
CONFIG_RELATIONSHIPS: str = "relationships.yaml"
CONFIG_RISK_RULES: str = "risk_rules.yaml"
CONFIG_VALIDATION: str = "validation.yaml"
CONFIG_EXPORT: str = "export.yaml"
