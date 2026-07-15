# System Architecture

## Overview

Early Stunting Risk AI is an **end-to-end AI system** built as a single unified service. Both the **frontend** and **backend** are served by a single FastAPI application, with Jinja2 server-side rendering for the web interface. This architecture simplifies deployment while maintaining a clean separation of concerns at the code level.

## High-Level Architecture

```
Browser
   │
   ▼
FastAPI Application (single service)
   ├── Jinja2 Templates + Static Files (Frontend)
   │      ├── index.html         → Landing page
   │      ├── prediction.html    → Prediction form
   │      ├── result.html        → Prediction result display
   │      ├── model.html         → Model information page
   │      └── about.html         → About page
   │
   └── API + ML Engine (Backend)
          ├── REST Endpoints
          ├── Predictor Service
          ├── Model Loader (Singleton)
          └── Explainability Service
                 │
                 ▼
          ML Artifacts (on disk)
           ├── best_pipeline.pkl    (CatBoost pipeline)
           ├── label_encoder.pkl
           ├── feature_names.json
           ├── feature_importance.csv (SHAP)
           └── explanation_summary.json
```

## Component Breakdown

### 1. Web Layer (Frontend)

Served directly by FastAPI using **Jinja2 templating engine**:

- **Templates**: HTML files with Tailwind CSS for responsive design, located at `backend/app/templates/`
- **Static Files**: CSS customizations, JavaScript, and images at `backend/app/static/`
- **No separate frontend framework** - server-side rendering keeps the stack simple and deployment-friendly

### 2. Service Layer (Backend)

Core business logic is organized into services:

| Service | File | Responsibility |
|---------|------|----------------|
| **Predictor** | `core/predictor.py` | Model inference, result formatting, recommendations |
| **Model Loader** | `core/model_loader.py` | Singleton - loads all ML artifacts at startup |
| **Explainability** | `core/explainability.py` | Read-only access to pre-computed SHAP results |

### 3. API Layer (REST Endpoints)

API routes are registered in `main.py` via sub-modules:

| Router Module | Endpoints |
|---------------|-----------|
| `api/prediction.py` | `POST /predict` - JSON-based prediction |
| `api/explainability.py` | `GET /feature-importance`, `GET /explanation-summary` |
| `api/model.py` | `GET /model-data` - model metadata |

Additional page routes are defined directly in `main.py`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/prediction` | Prediction input form |
| `POST` | `/predict-and-show` | Form submission → prediction → result page |
| `GET` | `/result` | Result display page |
| `GET` | `/model-info` | Model information |
| `GET` | `/about` | About the project |
| `GET` | `/health` | Health check for deployment monitoring |
| `GET` | `/api` | API metadata |

### 4. ML Layer (Model Artifacts)

Trained model artifacts are stored under `model/`:

```
model/
├── trained_models/
│   └── best_pipeline.pkl        # CatBoost pipeline (joblib)
├── artifacts/
│   ├── label_encoder.pkl         # Label encoder for risk_level
│   ├── feature_names.json        # 20 original feature names
│   └── training_config.json      # Training metadata
├── metrics/
│   ├── metrics.json              # Test set metrics
│   ├── cross_validation.json     # Cross-validation results
│   ├── classification_report.json
│   └── model_info.json           # Model metadata
└── explainability/
    ├── feature_importance.csv    # SHAP-based feature importance
    └── explanation_summary.json  # SHAP global explanation summary
```

## Request Flow (Prediction)

### Form-based flow (`/predict-and-show`):

```
User fills form → POST /predict-and-show
    │
    ▼
FastAPI receives Form data
    │
    ▼
PredictionRequest (Pydantic validation)
    │
    ▼
Predictor.predict(request)
    ├── Convert to DataFrame
    ├── model.predict_proba() → class probabilities
    ├── Decode prediction label (Low / Medium / High)
    ├── Calculate confidence score
    ├── Get top risk factors (from SHAP importance)
    └── Generate personalized recommendation
    │
    ▼
PredictionResponse → rendered in result.html
```

### JSON API flow (`POST /predict`):

```
Client → POST /predict (JSON body)
    │
    ▼
FastAPI validates → PredictionRequest schema
    │
    ▼
Predictor.predict(request)  (same logic)
    │
    ▼
JSON: PredictionResponse
```

## Design Patterns

1. **Singleton Pattern** - `ModelLoader` loads model artifacts once at startup and provides cached access throughout the application lifetime via `get_instance()`.
2. **Service Layer Pattern** - Predictor and Explainability services encapsulate business logic, keeping route handlers thin.
3. **Template Inheritance** - Jinja2 base templates with block overrides for consistent UI across pages.
4. **Lifespan Events** - FastAPI's `lifespan` context manager handles startup validation and shutdown cleanup.

## Configuration

All settings are managed through `app/config.py` using Pydantic Settings, supporting:

- Environment variables (via `.env` file)
- Cross-platform path resolution with `pathlib`
- Path validation at startup to catch missing artifacts early
