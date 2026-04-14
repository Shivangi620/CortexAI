# backend/core/utils.py

from infra.config import CONFIG


def clamp_confidence(value: float) -> float:
    """Clamps an AI confidence or similarity score to the max configured value to boost trust."""
    try:
        value = float(value)
    except Exception:
        value = 0.0

    max_conf = CONFIG.get("max_confidence", 100)

    try:
        max_conf = float(max_conf)
    except Exception:
        max_conf = 100.0

    return min(max_conf, value)