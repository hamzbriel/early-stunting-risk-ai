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

@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """
    Root endpoint returning application information.

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
