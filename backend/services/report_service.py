"""
services/report_service.py
Feature 10: Auto ML Report Generator — produces a professional PDF.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List


def generate_pdf(
    job_id: str,
    results: Dict[str, Any],
    profile: Dict[str, Any],
    insights: Dict[str, Any],
    story: str,
    recommendations: List[Dict[str, Any]],
) -> str:
    """
    Generate a multi-page PDF report and save it.
    Returns the path to the generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return _fallback_txt_report(job_id, results, profile)

    os.makedirs("tmp", exist_ok=True)
    pdf_path = f"tmp/{job_id}_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle("title",   parent=styles["Title"],   fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    h1_style      = ParagraphStyle("h1",      parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#16213e"), spaceAfter=4)
    h2_style      = ParagraphStyle("h2",      parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#0f3460"), spaceAfter=3)
    body_style    = ParagraphStyle("body",    parent=styles["Normal"],   fontSize=10, spaceAfter=4)
    caption_style = ParagraphStyle("caption", parent=styles["Normal"],   fontSize=8,  textColor=colors.grey)
    center_style  = ParagraphStyle("center",  parent=styles["Normal"],   alignment=TA_CENTER, fontSize=10)

    # Helper
    def hr():
        return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0"), spaceAfter=6)

    def spacer(h=0.3):
        return Spacer(1, h * cm)

    content = []

    # ─────────────────────────────────────────────────────────
    # PAGE 1: Title + Model Summary
    # ─────────────────────────────────────────────────────────
    best_model   = results.get("best_model", "Unknown")
    metric_name  = results.get("metric_name", "Score")
    score        = results.get("score", 0)
    is_clf       = results.get("is_classification", True)
    target       = results.get("target", "target")
    task_label   = "Classification" if is_clf else "Regression"

    content.append(Paragraph("AutoML Studio", title_style))
    content.append(Paragraph("Automated Machine Learning Report", h2_style))
    content.append(spacer(0.5))
    content.append(hr())

    summary_data = [
        ["Parameter", "Value"],
        ["Best Model",       best_model],
        ["Task Type",        task_label],
        ["Target Column",    target],
        ["Primary Metric",   metric_name],
        ["Score",            f"{score}%" if not (metric_name and 'R²' in metric_name) else f"{score/100:.3f}"],
        ["Job ID",           job_id[:16] + "..."],
    ]
    summary_table = Table(summary_data, colWidths=[6 * cm, 10 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 11),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f8f8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 1), (-1, -1), 10),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    content.append(summary_table)

    # ─────────────────────────────────────────────────────────
    # PAGE 2: Dataset Profile
    # ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("Dataset Profile", h1_style))
    content.append(hr())

    rows_count  = profile.get("rows", "N/A")
    cols_count  = profile.get("cols", "N/A")
    miss_pct    = profile.get("missing_pct", 0)
    imbalance   = profile.get("imbalance", "N/A")
    health      = profile.get("health_score", "N/A")

    ds_data = [
        ["Metric", "Value"],
        ["Total Rows",       str(rows_count)],
        ["Total Columns",    str(cols_count)],
        ["Missing Values",   f"{miss_pct:.1f}%" if isinstance(miss_pct, float) else str(miss_pct)],
        ["Class Imbalance",  str(imbalance)],
        ["Health Score",     str(health)],
    ]
    eda = results.get("eda_summary", {})
    if eda:
        ds_data.append(["Numeric Features",    str(eda.get("numeric_features", "N/A"))])
        ds_data.append(["Categorical Features", str(eda.get("categorical_features", "N/A"))])

    ds_table = Table(ds_data, colWidths=[6 * cm, 10 * cm])
    ds_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    content.append(ds_table)

    # ─────────────────────────────────────────────────────────
    # PAGE 3: Model Leaderboard
    # ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("Model Leaderboard", h1_style))
    content.append(hr())

    leaderboard = results.get("leaderboard", [])
    if leaderboard:
        lb_header = ["Rank", "Model", metric_name, "Phase"]
        if is_clf:
            lb_header += ["Precision", "Recall", "F1"]
        else:
            lb_header += ["MSE", "MAE"]

        lb_rows = [lb_header]
        for i, entry in enumerate(leaderboard[:8]):
            row = [
                str(i + 1),
                entry.get("model", ""),
                f"{entry.get('score', 0)}%",
                entry.get("phase", "").replace("_", " "),
            ]
            if is_clf:
                row += [
                    f"{entry.get('precision', '—')}%" if entry.get('precision') else "—",
                    f"{entry.get('recall', '—')}%" if entry.get('recall') else "—",
                    f"{entry.get('f1', '—')}%" if entry.get('f1') else "—",
                ]
            else:
                row += [
                    str(entry.get("mse", "—")),
                    str(entry.get("mae", "—")),
                ]
            lb_rows.append(row)

        col_w = [1.2 * cm, 4 * cm, 2.5 * cm, 3 * cm] + [2 * cm] * 3
        lb_table = Table(lb_rows, colWidths=col_w[:len(lb_header)])
        lb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#533483")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f0ff")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING",    (0, 0), (-1, -1), 5),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d4edda")),  # highlight winner
        ]))
        content.append(lb_table)
    else:
        content.append(Paragraph("No leaderboard data available.", body_style))

    # ─────────────────────────────────────────────────────────
    # PAGE 4: SHAP Feature Importance
    # ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("Feature Importance (SHAP)", h1_style))
    content.append(hr())

    shap_data = results.get("shap_summary", {})
    if shap_data:
        top_features = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
        total_importance = sum(abs(v) for _, v in top_features) or 1

        shap_rows = [["Feature", "SHAP Importance", "% Contribution"]]
        for feat, imp in top_features:
            pct = abs(imp) / total_importance * 100
            shap_rows.append([str(feat)[:30], f"{imp:.5f}", f"{pct:.1f}%"])

        shap_table = Table(shap_rows, colWidths=[8 * cm, 4 * cm, 4 * cm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff5f5")]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        content.append(shap_table)
        content.append(spacer())
        content.append(Paragraph(
            "SHAP (SHapley Additive exPlanations) values measure each feature's average "
            "contribution to model predictions. Higher absolute values = stronger influence.",
            caption_style,
        ))
    else:
        content.append(Paragraph("No SHAP data available. Re-train with Balanced or Full mode.", body_style))

    # ─────────────────────────────────────────────────────────
    # PAGE 5: Recommendations
    # ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("Recommendations", h1_style))
    content.append(hr())

    if recommendations:
        for rec in recommendations[:10]:
            priority = rec.get("priority", "")
            title    = rec.get("title", "")
            detail   = rec.get("detail", "")
            content.append(Paragraph(f"<b>{priority} — {title}</b>", body_style))
            content.append(Paragraph(detail, ParagraphStyle("indent", parent=body_style,
                                                             leftIndent=20, textColor=colors.HexColor("#555555"))))
            content.append(spacer(0.2))
    else:
        content.append(Paragraph("No recommendations generated.", body_style))

    # ─────────────────────────────────────────────────────────
    # PAGE 6: ELI5 Story + Conclusion
    # ─────────────────────────────────────────────────────────
    content.append(PageBreak())
    content.append(Paragraph("Summary & Insights", h1_style))
    content.append(hr())

    eli5 = insights.get("eli5", "") or ""
    if eli5:
        content.append(Paragraph("ELI5 Explanation", h2_style))
        content.append(Paragraph(eli5, body_style))
        content.append(spacer())

    if story:
        content.append(Paragraph("Training Story", h2_style))
        for line in story.split("\n")[:20]:
            if line.strip():
                content.append(Paragraph(line.strip(), body_style))
        content.append(spacer())

    content.append(hr())
    content.append(Paragraph(
        f"Generated by AutoML Studio V4 | Job: {job_id[:12]}... | "
        f"Model: {best_model} | {metric_name}: {score}%",
        caption_style,
    ))

    doc.build(content)
    return pdf_path


def _fallback_txt_report(job_id: str, results: Dict, profile: Dict) -> str:
    """Minimal text fallback if reportlab not installed."""
    path = f"tmp/{job_id}_report.txt"
    with open(path, "w") as f:
        f.write(f"AutoML Studio Report\nJob: {job_id}\n")
        f.write(f"Model: {results.get('best_model')}\n")
        f.write(f"Score: {results.get('score')}%\n")
        f.write(f"Rows: {profile.get('rows')}\n")
    return path
