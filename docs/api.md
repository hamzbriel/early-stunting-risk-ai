# API Documentation

## Overview

The FastAPI backend serves both the **web interface** (server-rendered HTML pages) and the **REST API** (JSON endpoints). The API is fully documented via Swagger at `/docs` and ReDoc at `/redoc` when the application is running.

**Base URLs**:

- Local: `http://localhost:8000`
- Production: `https://your-app.onrender.com`

## Page Endpoints

These endpoints serve HTML pages via Jinja2 templates and are primarily for browser use.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page with project overview |
| `GET` | `/prediction` | Prediction input form |
| `GET` | `/result` | Prediction result display page |
| `GET` | `/model-info` | Model details and performance metrics |
| `GET` | `/about` | Project information and tech stack |

## Form Prediction Endpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict-and-show` | Submit prediction form → rendered result page |

This endpoint accepts **form-encoded** data (not JSON) and renders the result using `result.html`.

## Health & Info Endpoints

### `GET /health`

Health check for deployment monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### `GET /api`

API metadata and version info.

**Response:**
```json
{
  "project": "Early Stunting Risk AI",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

## JSON API Endpoints

### `POST /predict`

Predict stunting risk level for a child. Accepts JSON with all required features and returns structured risk assessment.

**Request Body:**

```json
{
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
```

**Response:**

```json
{
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
  "recommendation": "Monitor child's growth regularly and ensure adequate protein intake. Increase protein-rich foods in child's diet."
}
```

**Field Details:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Risk level: `Low`, `Medium`, or `High` |
| `confidence` | float | Confidence score (0-100%) |
| `probabilities` | object | Probability distribution across all risk levels |
| `top_risk_factors` | array | Top contributing features ranked by SHAP importance (up to 5) |
| `recommendation` | string | Personalized health recommendation |

### `GET /model-data`

Get comprehensive model metadata including type, features, classes, and evaluation metrics.

**Response:**

```json
{
  "model_type": "Gradient Boosting",
  "model_name": "CatBoost",
  "num_features": 20,
  "num_classes": 3,
  "classes": ["High", "Low", "Medium"],
  "training_date": "11 Juli 2026",
  "metrics": {
    "accuracy": 0.906,
    "precision": 0.906,
    "recall": 0.906,
    "f1_score": 0.906
  }
}
```

### `GET /feature-importance`

Get global feature importance scores from pre-computed SHAP analysis.

**Response:**

```json
[
  {
    "feature": "num__clean_water",
    "importance": 0.765
  },
  {
    "feature": "num__diarrhea_history",
    "importance": 0.702
  }
]
```

### `GET /explanation-summary`

Get pre-computed SHAP explanation summary with global statistics and interpretation guidance.

**Response:**

```json
{
  "project": "early-stunting-risk-ai",
  "explainability_method": "SHAP",
  "model": "CatBoostClassifier",
  "explainer": "TreeExplainer",
  "samples_analyzed": 300,
  "top_features": [
    {
      "feature": "num__clean_water",
      "mean_abs_shap": 0.765
    }
  ]
}
```

## Input Validation

All input fields are validated using Pydantic schemas:

### Field Validation Rules

| Field | Type | Constraints |
|-------|------|-------------|
| `age_month` | integer | 0 - 60 |
| `gender` | string | `M` or `F` |
| `birth_weight` | float | 0.5 - 7.0 kg |
| `birth_length` | float | 25.0 - 65.0 cm |
| `mother_age` | integer | 15 - 60 |
| `mother_education` | string | `none`, `primary`, `secondary`, `higher` |
| `mother_working` | integer | 0 or 1 |
| `father_education` | string | `none`, `primary`, `secondary`, `higher` |
| `father_working` | integer | 0 or 1 |
| `family_income` | string | `very_low`, `low`, `medium`, `high` |
| `sanitation` | string | `poor`, `fair`, `good` |
| `clean_water` | integer | 0 or 1 |
| `electricity` | integer | 0 or 1 |
| `house_density` | string | `low`, `medium`, `high` |
| `exclusive_breastfeeding` | integer | 0 or 1 |
| `protein_intake` | string | `low`, `medium`, `high` |
| `vitamin_intake` | string | `low`, `medium`, `high` |
| `immunization` | string | `none`, `partial`, `complete` |
| `diarrhea_history` | integer | 0 or 1 |
| `healthcare_access` | string | `poor`, `fair`, `good` |

### Validation Errors

Invalid inputs return a **422 Unprocessable Entity** response:

```json
{
  "success": false,
  "message": "Validation error",
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "birth_weight"],
      "msg": "Value error, Birth weight should be between 0.5 and 7.0 kg. Please check the input value."
    }
  ]
}
```

## Error Handling

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 422 | Validation error - invalid input values |
| 500 | Internal server error - prediction failure |

All error responses follow a consistent format:

```json
{
  "detail": "Error description message"
}
```

## Usage Examples

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
result = response.json()
print(f"Risk Level: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1f}%")
print(f"Recommendation: {result['recommendation']}")
```
