"""api/routes/reports.py — PDF report generation (Feature 10)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json
import os

from infra.database import get_db, JobModel, DatasetModel
from infra.result_contract import normalize_results

router = APIRouter(prefix="/api", tags=["reports"])


def _model_card_html(job_id: str, results: dict, profile: dict, insights: dict, story: str) -> str:
    rows = profile.get("rows", "—")
    cols = profile.get("cols", "—")
    metric_name = results.get("metric_name", "Score")
    score = results.get("score", "—")
    tested_models = results.get("tested_models", [])
    top_rows = []
    for row in tested_models[:8]:
        top_rows.append(
            f"<tr><td>{row.get('model','—')}</td><td>{row.get('status','—')}</td><td>{row.get('sweep_score','—')}</td><td>{row.get('best_cv_score','—')}</td><td>{row.get('holdout_score','—')}</td></tr>"
        )
    tested_html = "".join(top_rows) or "<tr><td colspan='5'>No tested model details available.</td></tr>"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Model Card {job_id}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 32px; color: #1b1f23; background: #f7f4ee; }}
    h1, h2 {{ color: #0f3b57; }}
    .hero {{ background: linear-gradient(135deg, #fff4d6, #d9eef7); padding: 24px; border-radius: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin: 20px 0; }}
    .card {{ background: white; padding: 16px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    td, th {{ border-bottom: 1px solid #ddd; text-align: left; padding: 10px; }}
  </style>
</head>
<body>
  <div class='hero'>
    <h1>Model Card</h1>
    <p><strong>Model:</strong> {results.get("best_model","—")} | <strong>{metric_name}:</strong> {score}</p>
    <p>{story or insights.get("eli5","Auto-generated model card")}</p>
  </div>
  <div class='grid'>
    <div class='card'><h2>Intended Use</h2><p>Use for datasets matching the same feature semantics and preprocessing assumptions as training.</p></div>
    <div class='card'><h2>Dataset Snapshot</h2><p>Rows: {rows}<br/>Columns: {cols}<br/>Target: {results.get("target","—")}</p></div>
    <div class='card'><h2>Key Risks</h2><p>Distribution drift, missing-value changes, and class balance shifts may reduce quality.</p></div>
    <div class='card'><h2>Operational Notes</h2><p>Review drift checks and recalibrate thresholds for imbalanced production data.</p></div>
  </div>
  <h2>Tested Models</h2>
  <table>
    <tr><th>Model</th><th>Status</th><th>Sweep</th><th>Best CV</th><th>Holdout</th></tr>
    {tested_html}
  </table>
</body>
</html>"""


@router.get("/report/{job_id}/pdf")
def generate_pdf_report(job_id: str):
    """
    Feature 10: Auto ML Report Generator.
    Returns a 6-page PDF: summary, dataset profile, leaderboard, SHAP, recommendations, ELI5.
    """
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(
                status_code=409, detail=f"Job not completed (status: {job.status})"
            )

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}

        results = normalize_results(raw_results)

        try:
            insights = json.loads(job.insights_json) if job.insights_json else {}
        except Exception:
            insights = {}

        story = job.story or ""

        dataset = (
            db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()
        )

        try:
            profile = (
                json.loads(dataset.profile_json)
                if dataset and dataset.profile_json
                else {}
            )
        except Exception:
            profile = {}

    from core.recommendations import generate_recommendations

    recommendations = generate_recommendations(profile, results)

    from services.report_service import generate_pdf

    try:
        pdf_path = generate_pdf(
            job_id, results, profile, insights, story, recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    if not pdf_path or not isinstance(pdf_path, str) or not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF generation failed")

    return FileResponse(
        path=pdf_path,
        filename=f"automl_report_{job_id[:8]}.pdf",
        media_type="application/pdf",
    )


@router.get("/report/{job_id}/model-card")
def generate_model_card(job_id: str):
    with get_db() as db:
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=409, detail=f"Job not completed (status: {job.status})")

        try:
            raw_results = json.loads(job.results_json) if job.results_json else {}
        except Exception:
            raw_results = {}
        results = normalize_results(raw_results)

        try:
            insights = json.loads(job.insights_json) if job.insights_json else {}
        except Exception:
            insights = {}
        story = job.story or ""

        dataset = db.query(DatasetModel).filter(DatasetModel.id == job.dataset_id).first()
        try:
            profile = json.loads(dataset.profile_json) if dataset and dataset.profile_json else {}
        except Exception:
            profile = {}

    html = _model_card_html(job_id, results, profile, insights, story)
    out_dir = os.path.join("tmp", "model_cards")
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{job_id}_model_card.html")
    with open(html_path, "w") as f:
        f.write(html)
    return FileResponse(path=html_path, filename=f"model_card_{job_id[:8]}.html", media_type="text/html")
