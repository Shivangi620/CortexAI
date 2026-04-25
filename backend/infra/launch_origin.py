from __future__ import annotations


def parse_launch_origin(params: dict | None) -> dict:
    params = params or {}
    launch_context = params.get("launch_context") if isinstance(params, dict) else {}
    if not isinstance(launch_context, dict):
        launch_context = {}
    source = str(launch_context.get("source") or "manual").strip() or "manual"
    label = "Drift Reopen" if source == "drift_recommendation" else "Manual"
    return {
        "launch_source": source,
        "launch_label": label,
        "launch_context": launch_context,
    }

