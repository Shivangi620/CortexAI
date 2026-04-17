"""
AutoML Studio — Backend Entry Point (V4)
"""
import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from infra.logger import get_logger
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

log = get_logger(__name__)
WEB_DIR = Path(__file__).resolve().parent.parent / "frontend" / "web"


def _serve_web_file(filename: str):
    page_path = WEB_DIR / filename
    if page_path.exists():
        return FileResponse(page_path)
    return {"error": f"{filename} not found"}

csv.field_size_limit(int(1e9))

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return _serve_web_file("index.html")


@app.get("/results", include_in_schema=False)
def serve_results_console():
    return _serve_web_file("results.html")


@app.get("/results-console", include_in_schema=False)
def serve_results_console_alias():
    return serve_results_console()


@app.get("/dataset-dna", include_in_schema=False)
def serve_dataset_dna():
    return _serve_web_file("dataset-dna.html")


@app.get("/training-lab", include_in_schema=False)
def serve_training_lab():
    return _serve_web_file("training-lab.html")


@app.get("/experiment-tracker", include_in_schema=False)
def serve_experiment_tracker():
    return _serve_web_file("experiment-tracker.html")


@app.get("/drift-monitor", include_in_schema=False)
def serve_drift_monitor():
    return _serve_web_file("drift-monitor.html")


@app.get("/smart-ai-hub", include_in_schema=False)
def serve_smart_ai_hub():
    return _serve_web_file("smart-ai-hub.html")


# ── Health probe ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "version": "4.0.0"}


# ── Streamlit Reverse Proxy ──────────────────────────────────────────────────
# This catch-all route forwards any non-API requests to the Streamlit port (8001).
STREAMLIT_URL = "http://localhost:8001"

@app.api_route("/{path_name:path}", include_in_schema=False)
async def _proxy_to_streamlit(request: Request, path_name: str):
    # Skip if the path is explicitly an API route (though FastAPI should have caught it)
    if path_name.startswith("api/") or path_name.startswith("docs") or path_name.startswith("redoc") or path_name.startswith("openapi.json"):
        return {"error": "Not Found"}

    target_url = f"{STREAMLIT_URL}/{path_name}"
    if request.query_params:
        target_url += f"?{request.query_params}"

    async with httpx.AsyncClient() as client:
        # Standard proxy logic
        method = request.method
        headers = dict(request.headers)
        # We must remove host and adjust for the internal service
        headers.pop("host", None)
        
        # Handle Streamlit's specific requirements (like websockets if needed, but simple HTTP first)
        response = await client.request(
            method,
            target_url,
            content=await request.body(),
            headers=headers,
            timeout=60.0,
            follow_redirects=True,
        )

        return StreamingResponse(
            response.aiter_bytes(),
            status_code=response.status_code,
            headers=dict(response.headers),
        )


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        log.error(f"Failed to start server: {e}")
