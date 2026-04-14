def format_metric_value(metric_name, score):
    if score is None:
        return "—"
    name = (metric_name or "").lower()
    try:
        s = float(score)
    except (TypeError, ValueError):
        return str(score)

    if "r²" in name or name.strip() in {"r2", "r2 score"}:
        return f"{s/100:.3f}"
    if "rmse" in name or "mse" in name or "mae" in name:
        return f"{s:.4f}"
    return f"{s:.1f}%"

