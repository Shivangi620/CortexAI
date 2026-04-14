from infra.config import CONFIG


def compute_health_score(profile: dict) -> dict:
    score = 100
    issues = []
    bonuses = []

    # ✅ FIX 1: safe profile fallback
    profile = profile or {}

    # ── 1. Missing Values ─────────────────────────────────────────
    missing_pct = profile.get("missing_pct", 0) or 0

    # ✅ FIX 2: safe config access
    high_missing_thresh = CONFIG.get("high_missing", 10.0)

    if missing_pct > high_missing_thresh * 2:
        score -= 25
        issues.append(f"🔴 Severe missing data: {missing_pct}% of all cells are empty")
    elif missing_pct > high_missing_thresh:
        score -= 15
        issues.append(f"🟡 High missing data: {missing_pct}% — median imputation applied")
    elif missing_pct > 2:
        score -= 5
        issues.append(f"🟢 Minor missing data: {missing_pct}% — handled automatically")
    elif missing_pct == 0:
        score += 5
        bonuses.append("✅ Pristine data: No missing values detected")
    else:
        bonuses.append("✅ No significant missing values")

    # ── 2. Class Imbalance ────────────────────────────────────────
    if profile.get("imbalance") == "High ⚠️":
        score -= 20
        issues.append("🔴 Severe class imbalance — minority class model recall will suffer")
    else:
        score += 5
        bonuses.append("✅ Class distribution is perfectly balanced")

    # ── 3. Dataset Size ───────────────────────────────────────────
    rows = profile.get("rows", 0) or 0
    cols = profile.get("cols", 0) or 0

    # ✅ FIX 3: safe config key access
    small_limit = CONFIG.get("small_data_rows", 1000)

    if rows < 100:
        score -= 25
        issues.append(f"🔴 Critically small: only {rows} rows — model may memorize patterns (overfit) rather than learn to generalize.")
    elif rows < small_limit:
        score -= 15
        feat_msg = " and very few features" if cols < 3 else ""
        issues.append(f"⚠️ High Overfitting Risk: The dataset is small ({rows} rows){feat_msg}, so the model may memorize patterns instead of generalizing well to new data.")
    elif rows < 1000:
        score -= 5
        issues.append(f"🟡 Borderline size: {rows} rows — a slightly larger dataset would reduce the risk of overfitting.")
    elif rows >= 10_000:
        score += 5
        bonuses.append(f"✅ Large dataset: {rows:,} rows — great for complex models")
    else:
        bonuses.append(f"✅ Adequate dataset size: {rows:,} rows")

    # ── 4. Feature Quality ─────────────────────────────────────────
    col_stats = profile.get("column_stats", {}) or {}

    # ✅ FIX 4: ensure dict
    if not isinstance(col_stats, dict):
        col_stats = {}

    def safe_float(x, default=0):
        try:
            return float(x)
        except Exception:
            return default

    high_missing_cols = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and s.get("missing_pct", 0) > 30
    ]

    high_skew_cols = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and abs(safe_float(s.get("skew"))) > 5
    ]

    zero_var_cols = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and s.get("unique", 99) <= 1
    ]

    high_outlier_cols = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and s.get("outlier_pct", 0) > 5.0
    ]

    id_cols = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and s.get("semantic_type", "") == "ID/Index"
    ]

    fq_penalty = 0

    if high_missing_cols:
        pen = min(10, len(high_missing_cols) * 3)
        fq_penalty += pen
        issues.append(f"🔴 {len(high_missing_cols)} column(s) with >30% missing: {', '.join(high_missing_cols[:4])}")

    if high_skew_cols:
        pen = min(8, len(high_skew_cols) * 2)
        fq_penalty += pen
        issues.append(f"🟡 {len(high_skew_cols)} highly skewed feature(s): {', '.join(high_skew_cols[:4])} — consider log-transform")

    if zero_var_cols:
        fq_penalty += len(zero_var_cols) * 2
        issues.append(f"🔴 {len(zero_var_cols)} constant column(s) (zero variance): {', '.join(zero_var_cols[:4])} — safe to drop")

    if high_outlier_cols:
        pen = min(10, len(high_outlier_cols) * 2)
        fq_penalty += pen
        issues.append(f"🟡 {len(high_outlier_cols)} column(s) with significant outliers (>5%): {', '.join(high_outlier_cols[:4])} — models may be sensitive to these.")

    if id_cols:
        fq_penalty += len(id_cols) * 2
        issues.append(f"🟡 {len(id_cols)} ID/Index column(s) detected: {', '.join(id_cols[:4])} — these should be removed to prevent overfitting.")

    score -= min(20, fq_penalty)

    if fq_penalty == 0:
        score += 5
        bonuses.append("✅ All columns have good validity, distinctiveness, and low skew")

    # ── 5. Feature Diversity ─────────────────────────────────────
    num_cols = profile.get("num_cols") or []
    cat_cols = profile.get("cat_cols") or []

    if cols < 3:
        score -= 5
        issues.append("🟡 Very few features — model relationships are simple (may favour linear models)")
    elif cols > 100:
        score -= 5
        issues.append(f"🟡 High dimensionality: {cols} features — consider PCA or selection")
    elif num_cols and cat_cols:
        bonuses.append("✅ Healthy mix of numerical and categorical features")

    # ── Final clamp ────────────────────────────────────────────────
    score = max(0, min(100, score))

    if issues and score > 90:
        score = 90

    if score >= 95:
        grade = "A+"
    elif score >= 90:
        grade = "A"
        if issues:
            grade = "A-"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    color = (
        "#22c55e" if score >= 80 else
        "#eab308" if score >= 60 else
        "#f97316" if score >= 40 else
        "#ef4444"
    )

    return {
        "score": score,
        "grade": grade,
        "color": color,
        "issues": issues,
        "bonuses": bonuses,
        "summary": (
            f"Your dataset scores {score}/100 (Grade {grade}). "
            + (f"Found {len(issues)} issue(s) to address." if issues
               else "Dataset looks healthy!")
        ),
    }