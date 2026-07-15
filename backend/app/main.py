"""
FastAPI application entry point for Early Stunting Risk AI backend.

This module initializes the FastAPI application, configures middleware,
sets up startup/shutdown events, and registers all API routes.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.core.model_loader import ModelLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Startup:
        - Validate configuration paths
        - Load model artifacts
        - Initialize services

    Shutdown:
        - Cleanup resources if needed
    """
    # Startup
    print("\n" + "="*60)
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print("="*60 + "\n")

    # Validate paths
    print("Validating configuration paths...")
    missing_paths = settings.get_missing_paths()
    if missing_paths:
        print(f"WARNING: Missing {len(missing_paths)} artifacts:")
        for path in missing_paths:
            print(f"  - {path}")
    else:
        print("All required artifacts found!")

    # Load model artifacts using ModelLoader singleton
    print("\nLoading model artifacts...")
    try:
        loader = ModelLoader.get_instance()
        print(f"\nModel info: {loader.get_model_info()}")
    except Exception as e:
        print(f"\nERROR: Failed to load model artifacts: {e}")
        print("Application startup aborted.")
        raise

    print("\n" + "="*60)
    print(f"{settings.app_name} is ready!")
    print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    print("="*60 + "\n")

    yield

    # Shutdown
    print("\nShutting down application...")
    print("Cleanup complete.")


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Early Stunting Risk AI",
        "url": "https://github.com/yourusername/early-stunting-risk-ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with consistent format."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Validation error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected errors with consistent format."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": str(exc),
        },
    )


# Root Endpoints

@app.get("/", tags=["Pages"], include_in_schema=False)
async def home(request: Request):
    """
    Homepage - Landing page with project overview and features.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/prediction", tags=["Pages"], include_in_schema=False)
async def prediction_page(request: Request):
    """
    Prediction form page - Input form for stunting risk prediction.
    """
    return templates.TemplateResponse("prediction.html", {"request": request})


@app.get("/model-info", tags=["Pages"], include_in_schema=False)
async def model_info_page(request: Request):
    """
    Model information page - Displays model metadata and metrics.
    """
    return templates.TemplateResponse("model.html", {"request": request})


@app.get("/about", tags=["Pages"], include_in_schema=False)
async def about_page(request: Request):
    """
    About page - Project information and tech stack.
    """
    return templates.TemplateResponse("about.html", {"request": request})


from types import SimpleNamespace
from fastapi import FastAPI, Request, status, Form
from fastapi.responses import JSONResponse, RedirectResponse


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url=app.url_path_for("static", path="images/logo.png"))


@app.get("/result", tags=["Pages"], include_in_schema=False)
async def result_page(request: Request):
    """
    Result page - Displays prediction results.
    """
    return templates.TemplateResponse("result.html", {"request": request, "result": None})

@app.post("/predict-and-show", tags=["Prediction"], include_in_schema=False)
async def predict_and_show(request: Request,
                           age_month: int = Form(...),
                           gender: str = Form(...),
                           birth_weight: float = Form(...),
                           birth_length: float = Form(...),
                           mother_age: int = Form(...),
                           mother_education: str = Form(...),
                           mother_working: int = Form(...),
                           father_education: str = Form(...),
                           father_working: int = Form(...),
                           family_income: str = Form(...),
                           sanitation: str = Form(...),
                           clean_water: int = Form(...),
                           electricity: int = Form(...),
                           house_density: str = Form(...),
                           exclusive_breastfeeding: int = Form(...),
                           protein_intake: str = Form(...),
                           vitamin_intake: str = Form(...),
                           immunization: str = Form(...),
                           diarrhea_history: int = Form(...),
                           healthcare_access: str = Form(...)):
    """
    Handle form submission, perform prediction, and render result page.
    """
    from app.schemas.request import PredictionRequest
    from app.core.predictor import get_predictor

    pred_request = PredictionRequest(
        age_month=age_month,
        gender=gender,
        birth_weight=birth_weight,
        birth_length=birth_length,
        mother_age=mother_age,
        mother_education=mother_education,
        mother_working=mother_working,
        father_education=father_education,
        father_working=father_working,
        family_income=family_income,
        sanitation=sanitation,
        clean_water=clean_water,
        electricity=electricity,
        house_density=house_density,
        exclusive_breastfeeding=exclusive_breastfeeding,
        protein_intake=protein_intake,
        vitamin_intake=vitamin_intake,
        immunization=immunization,
        diarrhea_history=diarrhea_history,
        healthcare_access=healthcare_access
    )

    predictor = get_predictor()
    result = predictor.predict(pred_request)
    print(f"DEBUG: Prediksi berhasil untuk usia {pred_request.age_month}")

    return templates.TemplateResponse("result.html", {"request": request, "result": result})


@app.get("/api", tags=["Root"])
async def api_info() -> dict[str, Any]:
    """
    API information endpoint.

    Returns basic metadata about the API.
    """
    return {
        "project": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """
    Health check endpoint for deployment monitoring.

    Used by Render and other deployment platforms to verify
    that the application is running correctly.
    """
    return {"status": "healthy"}


# API Routes Registration
from app.api import explainability, model, prediction

app.include_router(model.router, tags=["Model"])
app.include_router(explainability.router, tags=["Explainability"])
app.include_router(prediction.router, tags=["Prediction"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
