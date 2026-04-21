"""
AutoML Studio — Backend Entry Point (V4)
"""
import csv
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from infra.logger import get_logger

log = get_logger(__name__)
csv.field_size_limit(int(1e9))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ── Route imports ─────────────────────────────────────────────────────────────

from api.routes.datasets import router as datasets_router
from api.routes.training import router as training_router
from api.routes.experiments import router as experiments_router
from api.routes.explain import router as explain_router
from api.routes.predict import router as predict_router
from api.routes.drift import router as drift_router
from api.routes.reports import router as reports_router
from api.routes.misc import router as misc_router


# ── Lifespan management ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    run_cleanup = os.getenv("AUTOML_STARTUP_CLEANUP", "false").lower() == "true"
    if run_cleanup:
        try:
            from infra.storage import cleanup_old_runs
            removed = cleanup_old_runs(days=7)
            if removed:
                log.info(f"Cleaned {removed} old artifact(s).")
        except Exception as e:
            log.warning(f"Cleanup skipped: {e}")
    yield


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoML Studio API V4",
    description="Production-grade AutoML backend with modular route architecture.",
    version="4.0.0",
    lifespan=lifespan,
)

APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend" / "static"

allowed_origins_raw = os.getenv("AUTOML_ALLOWED_ORIGINS", "*").strip()
allowed_origins = (
    ["http://localhost:3000", "http://localhost:8501"]
    if not allowed_origins_raw or allowed_origins_raw == "*"
    else [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount routers ─────────────────────────────────────────────────────────────
for r in [
    datasets_router,
    training_router,
    experiments_router,
    explain_router,
    predict_router,
    drift_router,
    reports_router,
    misc_router,
]:
    app.include_router(r)





# ── Health probe ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "4.0.0"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend-static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/index.html", include_in_schema=False)
    def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend_routes(full_path: str):
        if not full_path:
            return FileResponse(FRONTEND_DIR / "index.html")
        blocked_prefixes = ("api", "docs", "redoc", "openapi.json", "static")
        if full_path.startswith(blocked_prefixes):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = FRONTEND_DIR / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(FRONTEND_DIR / "index.html")





if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        log.error(f"Failed to start server: {e}")
