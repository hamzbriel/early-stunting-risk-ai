# Deployment

## Overview

The application is deployed on **Render** as a **single unified service** - the FastAPI backend serves both the API and the frontend (Jinja2 templates), eliminating the need for separate frontend hosting.

## Prerequisites

- **Python 3.12** (model artifacts are not compatible with newer Python/dependency versions)
- A [Render](https://render.com) account
- Git repository with the project pushed to GitHub/GitLab

## Local Development

### Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset (optional - pre-generated data available)
cd synthetic_data
python src/main.py
cd ..

# Run the API server
cd backend
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000` to access the application.

### Environment Variables

Configuration is handled through `backend/app/config.py` using Pydantic Settings. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `RELOAD` | `False` | Auto-reload on code changes (dev only) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Settings can be overridden via a `.env` file or environment variables.

## Deployment to Render

### Step 1: Choose Deployment Method

Render supports two approaches:

#### Option A: Render Blueprint (recommended)

Create a `render.yaml` file in the project root:

```yaml
services:
  - type: web
    name: early-stunting-risk-ai
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

#### Option B: Manual Web Service

1. In the Render Dashboard, click **New +** → **Web Service**
2. Connect your GitHub/GitLab repository
3. Configure:
   - **Name**: `early-stunting-risk-ai`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free or Starter

### Step 2: Enable Auto-Deploy

Render automatically deploys when changes are pushed to the connected branch (usually `main`).

## Application Structure on Render

```
/opt/render/project/src/
├── backend/
│   └── app/
│       ├── main.py            ← FastAPI entry point
│       ├── config.py          ← Settings
│       ├── api/               ← REST endpoints
│       ├── core/              ← Services (predictor, model loader)
│       ├── schemas/           ← Pydantic models
│       ├── templates/         ← Jinja2 HTML templates
│       └── static/            ← CSS, JS, images
├── model/
│   ├── trained_models/        ← best_pipeline.pkl
│   ├── artifacts/             ← encoders, configs
│   ├── metrics/               ← evaluation results
│   └── explainability/       ← SHAP analysis
├── requirements.txt
└── pyproject.toml
```

## Dependencies

All Python dependencies are specified in `requirements.txt`:

| Category | Key Packages |
|----------|-------------|
| **Backend** | fastapi, uvicorn, jinja2, python-multipart |
| **Machine Learning** | scikit-learn==1.6.1, catboost, xgboost, lightgbm |
| **Explainability** | shap |
| **Data Processing** | numpy, pandas, joblib |
| **Configuration** | pydantic, pydantic-settings, pyyaml |

> **⚠️ Important:** The serialized model (`best_pipeline.pkl`) was trained with `scikit-learn==1.6.1`. Loading it with a different version can cause unpickling errors. Ensure compatibility when updating dependencies.

## Health Checks

The deployment platform can monitor application health using the `/health` endpoint:

```json
{
  "status": "healthy"
}
```

Render automatically pings this endpoint to ensure the service is running. On the Free tier, the service spins down after 15 minutes of inactivity and spins up on the next request (may cause a 30–60 second delay).

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Model fails to load** | Ensure `best_pipeline.pkl` is committed to Git (not in `.gitignore`) |
| **scikit-learn version mismatch** | Verify `scikit-learn==1.6.1` in `requirements.txt` |
| **Start command fails** | Check that `cd backend && uvicorn app.main:app ...` points to the correct path |
| **Static files not loading** | Verify `app.mount("/static", ...)` path resolves correctly in production |
| **Deploy takes too long** | Initial deploy downloads all dependencies - subsequent deploys use cache |

### Logs

View real-time logs in the Render Dashboard under your service's **Logs** tab to debug startup issues or runtime errors.
