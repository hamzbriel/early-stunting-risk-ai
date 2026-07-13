# Backend - Early Stunting Risk AI

FastAPI backend untuk sistem prediksi risiko stunting menggunakan Machine Learning.

## Overview

Backend ini menyediakan REST API untuk melakukan inferensi model Machine Learning yang telah dilatih sebelumnya. Backend **tidak melakukan training ulang** dan hanya menggunakan artifact model yang sudah tersedia.

## Architecture

Backend dibangun dengan prinsip **clean architecture** dan **separation of concerns**:

```
backend/
├── app/
│   ├── __init__.py           # Application package initialization
│   ├── main.py               # FastAPI application entry point
│   ├── config.py             # Configuration management (paths, settings)
│   ├── dependencies.py       # Dependency injection
│   │
│   ├── api/                  # API endpoints (route handlers)
│   │   ├── __init__.py
│   │   ├── health.py         # Health check endpoint
│   │   ├── prediction.py     # Prediction endpoints
│   │   ├── model.py          # Model info endpoints
│   │   └── explainability.py # Explainability endpoints
│   │
│   ├── core/                 # Business logic
│   │   ├── __init__.py
│   │   ├── model_loader.py   # Load ML model & artifacts (singleton)
│   │   ├── predictor.py      # Prediction service
│   │   ├── explainability.py # Explainability service
│   │   ├── validation.py     # Input validation logic
│   │   └── logger.py         # Logging configuration
│   │
│   ├── schemas/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── request.py        # Request schemas
│   │   ├── response.py       # Response schemas
│   │   ├── prediction.py     # Prediction-specific schemas
│   │   └── model_info.py     # Model info schemas
│   │
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # Data preprocessing utilities
│   │   ├── file_utils.py     # File handling utilities
│   │   └── helper.py         # General helper functions
│   │
│   ├── static/               # Static files (CSS, JS, images)
│   └── templates/            # Jinja2 templates (HTML)
│
└── requirements.txt          # Python dependencies
```

## Design Principles

1. **Modular**: Setiap modul memiliki tanggung jawab yang jelas
2. **Scalable**: Mudah untuk menambahkan fitur baru
3. **Maintainable**: Kode yang bersih dan terdokumentasi
4. **Type-safe**: Menggunakan type hints dan Pydantic validation
5. **Performance**: Model dimuat sekali saat startup, bukan per request

## Tech Stack

- **Framework**: FastAPI 0.115.0
- **Server**: Uvicorn 0.32.0
- **Validation**: Pydantic 2.9.2
- **ML Libraries**: scikit-learn, pandas, numpy, joblib
- **Explainability**: SHAP 0.46.0
- **Template Engine**: Jinja2 3.1.4

## Dependencies

Lihat `requirements.txt` untuk daftar lengkap dependencies.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Application

(Will be added in next sections)

## API Documentation

FastAPI menyediakan dokumentasi interaktif otomatis:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing

(Will be added in testing section)

## Model Artifacts

Backend menggunakan artifact dari fase Machine Learning sebelumnya:

```
../model/
├── trained_models/
│   └── best_pipeline.pkl      # Trained model
├── artifacts/
│   ├── label_encoder.pkl      # Label encoder
│   ├── feature_names.json     # Feature names
│   ├── training_config.json   # Training configuration
│   └── evaluation_results.json # Evaluation metrics
└── explainability/
    ├── feature_importance.csv  # Feature importance scores
    └── explanation_summary.json # SHAP summary
```

## Security Notes

- Input validation menggunakan Pydantic
- Error handling yang konsisten
- Logging untuk monitoring dan debugging

---

**Note**: Backend ini adalah bagian dari proyek end-to-end AI untuk prediksi risiko stunting. Lihat repository root untuk dokumentasi lengkap.
