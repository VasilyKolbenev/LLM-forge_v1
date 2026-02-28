"""Unified FastAPI application for llm-forge Web UI.

Serves both the REST API and static React frontend.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from llm_forge.ui.routes import training, datasets, experiments, evaluation
from llm_forge.ui.routes import export_routes, hardware
from llm_forge.ui.assistant import router as assistant_router
from llm_forge.ui.metrics import router as metrics_router
from llm_forge.ui.routes.compute import router as compute_router
from llm_forge.ui.routes.workflows import router as workflows_router
from llm_forge.ui.routes.prompts import router as prompts_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app with all routes mounted.
    """
    app = FastAPI(
        title="llm-forge",
        description="LLM fine-tuning dashboard",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(training.router, prefix="/api/v1")
    app.include_router(datasets.router, prefix="/api/v1")
    app.include_router(experiments.router, prefix="/api/v1")
    app.include_router(evaluation.router, prefix="/api/v1")
    app.include_router(export_routes.router, prefix="/api/v1")
    app.include_router(hardware.router, prefix="/api/v1")
    app.include_router(assistant_router, prefix="/api/v1")
    app.include_router(metrics_router, prefix="/api/v1")
    app.include_router(compute_router, prefix="/api/v1")
    app.include_router(workflows_router, prefix="/api/v1")
    app.include_router(prompts_router, prefix="/api/v1")

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"status": "ok"}

    # Serve React static build if available
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True))

    return app


def start_ui_server(host: str = "0.0.0.0", port: int = 8888) -> None:
    """Start the UI server with uvicorn.

    Args:
        host: Server host.
        port: Server port.
    """
    import uvicorn

    app = create_app()
    logger.info("Starting llm-forge UI on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
