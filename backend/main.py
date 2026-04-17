"""
AutoML Studio — Backend Entry Point (V4)
"""
import csv
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from infra.logger import get_logger

log = get_logger(__name__)
csv.field_size_limit(int(1e9))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ── Route imports ─────────────────────────────────────────────────────────────
try:
    from api.routes.datasets import router as datasets_router
except Exception:
    datasets_router = None

try:
    from api.routes.training import router as training_router
except Exception:
    training_router = None

try:
    from api.routes.experiments import router as experiments_router
except Exception:
    experiments_router = None

try:
    from api.routes.explain import router as explain_router
except Exception:
    explain_router = None

try:
    from api.routes.predict import router as predict_router
except Exception:
    predict_router = None

try:
    from api.routes.drift import router as drift_router
except Exception:
    drift_router = None

try:
    from api.routes.reports import router as reports_router
except Exception:
    reports_router = None

try:
    from api.routes.misc import router as misc_router
except Exception:
    misc_router = None


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

allowed_origins_raw = os.getenv("AUTOML_ALLOWED_ORIGINS", "*").strip()
allowed_origins = (
    ["*"]
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


# ── Mount routers safely ──────────────────────────────────────────────────────
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
    try:
        if r:
            app.include_router(r)
    except Exception as e:
        log.warning(f"Router mount failed: {e}")





# ── Health probe ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "4.0.0"}





if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        log.error(f"Failed to start server: {e}")
