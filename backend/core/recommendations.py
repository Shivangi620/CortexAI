# ── Smart Recommendations ─────────────────────────────────────────────────────

def generate_recommendations(profile: dict, results: dict) -> list:
    """
    Return a prioritised list of actionable data science recommendations.
    Each item: {priority, icon, title, detail, action}
    """
    recs = []

    # ✅ FIX 1: safe defaults
    col_stats = profile.get("column_stats") or {}
    shap = results.get("shap_summary") or {}
    is_clf = results.get("is_classification", True)

    # ✅ FIX 2: safe score
    try:
        score = float(results.get("score", 0))
    except Exception:
        score = 0

    rows = profile.get("rows", 0)

    # ── Missing values ────────────────────────────────────────────────────
    for col, stats in col_stats.items():
        if not isinstance(stats, dict):
            continue

        mp = stats.get("missing_pct", 0)
        if mp > 50:
            recs.append({
                "priority": "🔴 Critical",
                "title": f"Drop column '{col}'",
                "detail": f"{mp}% of values are missing — this column adds more noise than signal.",
                "action": f"df.drop(columns=['{col}'], inplace=True)",
            })
        elif mp > 20:
            recs.append({
                "priority": "🟡 Important",
                "title": f"Review imputation for '{col}'",
                "detail": f"{mp}% missing. Consider domain-specific fills or median/mode by group.",
                "action": f"df['{col}'].fillna(df['{col}'].median(), inplace=True)",
            })

    # ── Zero variance ─────────────────────────────────────────────────────
    zero_var = [
        c for c, s in col_stats.items()
        if isinstance(s, dict) and s.get("unique", 99) <= 1
    ]
    for col in zero_var:
        recs.append({
            "priority": "🔴 Critical",
            "title": f"Drop constant column '{col}'",
            "detail": "This column has only one unique value and contributes nothing to the model.",
            "action": f"df.drop(columns=['{col}'], inplace=True)",
        })

    # ── High skew ─────────────────────────────────────────────────────────
    for col, stats in col_stats.items():
        if not isinstance(stats, dict):
            continue

        skew = stats.get("skew")
        if skew is not None:
            try:
                skew_val = float(skew)
                if abs(skew_val) > 5:
                    recs.append({
                        "priority": "🟡 Important",
                        "title": f"Log-transform skewed feature '{col}'",
                        "detail": f"Skewness = {skew_val:.2f}. Heavy skew hurts linear models and slows tree convergence.",
                        "action": f"import numpy as np\ndf['{col}'] = np.log1p(df['{col}'].clip(lower=0))",
                    })
            except (ValueError, TypeError):
                continue

    # ── Class imbalance ───────────────────────────────────────────────────
    if profile.get("imbalance") == "High ⚠️" and is_clf:
        recs.append({
            "priority": "🔴 Critical",
            "title": "Address class imbalance with SMOTE",
            "detail": "Your target classes are severely skewed. Minority class recall will be poor.",
            "action": "from imblearn.over_sampling import SMOTE\nX_res, y_res = SMOTE().fit_resample(X, y)",
        })

    # ── Small dataset ─────────────────────────────────────────────────────
    if rows < 500:
        recs.append({
            "priority": "🟡 Important",
            "title": "Expand your dataset with Synthetic Data",
            "detail": f"Only {rows} rows detected. Use the Synthetic Generator on the AI Tools page.",
            "action": "Go to 8_AI_Tools → Synthetic Data Generator",
        })

    # ── Low score suggestions ─────────────────────────────────────────────
    if score < 70:
        recs.append({
            "priority": "🟡 Important",
            "title": "Try feature engineering",
            "detail": f"Score is {score}%. Consider creating interaction terms or polynomial features.",
            "action": "from sklearn.preprocessing import PolynomialFeatures\npf = PolynomialFeatures(degree=2, interaction_only=True)",
        })

        if rows < 2000:
            recs.append({
                "priority": "🟡 Important",
                "title": "Collect more training data",
                "detail": "Score under 70% with a small dataset usually means the model is data-hungry.",
                "action": "Focus data collection on the features with highest SHAP importance.",
            })

    # ── SHAP-based feature pruning ────────────────────────────────────────
    if isinstance(shap, dict) and shap:
        num_cols = profile.get("num_cols") or []
        cat_cols = profile.get("cat_cols") or []

        all_features = set(num_cols + cat_cols)
        important_features = set(shap.keys())
        unimportant = all_features - important_features

        if len(unimportant) >= 3:
            sample = list(unimportant)[:5]
            recs.append({
                "priority": "🟢 Suggestion",
                "title": f"Consider dropping {len(unimportant)} low-importance features",
                "detail": f"SHAP analysis shows these features contribute minimally: {', '.join(sample)}",
                "action": f"df.drop(columns={sample[:3]}, inplace=True)",
            })

    # ── Positive confirmation ────────────────────────────────────────────
    if score >= 85 and not recs:
        recs.append({
            "priority": "✅ Excellent",
            "title": "Your pipeline is well-optimised!",
            "detail": f"Score of {score}% with a clean dataset. Consider deploying the model.",
            "action": "Download the export bundle and run the generated API.",
        })

    # Sort: Critical first
    order = {"🔴 Critical": 0, "🟡 Important": 1, "🟢 Suggestion": 2, "✅ Excellent": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 9))

    return recs