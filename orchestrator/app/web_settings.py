"""
Production web settings, controlled by environment variables (safe by default):

- CORS_ALLOWED_ORIGINS: comma-separated browser origins allowed to call the API,
  e.g. "https://wardmitra.example.in,https://admin.wardmitra.example.in".
  Empty (default) = no cross-origin browser access. "*" = any origin (local testing only).
  Native mobile apps and server-to-server calls are not affected by CORS.
- ENABLE_TEST_UI: "true" serves the testing studio at /ui, /studio and /static. Default off.
"""
import os
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

TRUE_VALUES = {"1", "true", "yes", "on"}


def cors_allowed_origins() -> List[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def test_ui_enabled() -> bool:
    return os.getenv("ENABLE_TEST_UI", "false").strip().lower() in TRUE_VALUES


def configure_cors(app: FastAPI) -> List[str]:
    """
    Allow only the configured origins. Auth is a bearer token, not cookies, so credentials are off;
    the Authorization header is allowed explicitly for the citizen JWT.
    """
    origins = cors_allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )
    logger.info(f"CORS allowed origins: {origins or 'none (cross-origin browser calls blocked)'}")
    return origins


def register_test_ui(app: FastAPI, static_dir: Path) -> bool:
    """Serve the testing studio only when ENABLE_TEST_UI is set; otherwise the routes do not exist (404)."""
    if not test_ui_enabled():
        logger.info("Testing studio (/ui) disabled. Set ENABLE_TEST_UI=true to serve it locally.")
        return False

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/ui", include_in_schema=False)
    @app.get("/studio", include_in_schema=False)
    def serve_ui():
        """Serve the interactive WardMitra AI Testing Studio."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return RedirectResponse(url="/docs")

    @app.get("/", include_in_schema=False)
    def redirect_to_ui():
        """Redirect root to testing studio UI."""
        return RedirectResponse(url="/ui")

    logger.warning("Testing studio is ENABLED at /ui — do not enable this in production.")
    return True
