def generate_pipeline_graph(profile: dict, results: dict) -> str:
    """Generate Mermaid.js flowchart of the AutoML pipeline."""

    # ✅ FIX 1: safe defaults if None passed
    profile = profile or {}
    results = results or {}

    n_rows = profile.get("rows", 0) or 0
    n_cols = profile.get("cols", 0) or 0

    num_c = len(profile.get("num_cols") or [])
    cat_c = len(profile.get("cat_cols") or [])

    best = results.get("best_model", "Model") or "Model"
    score = results.get("score", 0) or 0
    metric = results.get("metric_name", "Score") or "Score"

    leaderboard = results.get("leaderboard") or []
    has_shap = bool(results.get("shap_summary"))

    # ✅ FIX 2: safe leaderboard access (prevent KeyError / TypeError)
    model_nodes = ""
    for i, e in enumerate(leaderboard[:5]):
        try:
            model_name = str(e.get("model", "Model"))
            model_score = e.get("score", 0)
            star = "⭐" if model_name == best else ""
            model_nodes += (
                f'    Arena --> M{i}["{model_name}\\n{model_score}%{star}"]\n'
            )
        except Exception:
            continue  # skip bad entries safely

    shap_node = (
        '    Best --> SHAP["🧭 SHAP\\nFeature Importance"]\n'
        if has_shap else ""
    )

    # ✅ FIX 3: ensure formatting doesn't break if values are weird types
    try:
        rows_fmt = f"{int(n_rows):,}"
    except Exception:
        rows_fmt = "0"

    try:
        cols_fmt = f"{int(n_cols)}"
    except Exception:
        cols_fmt = "0"

    return f"""flowchart TD
    Raw["📂 Raw Data\\n{rows_fmt} rows x {cols_fmt} cols"]
    Raw --> Loader["🔌 Universal File Loader"]
    Loader --> Profile["🧬 Data Profiler"]
    Profile --> Pre["⚙️ Preprocessing"]
    Pre --> Num["🔢 Numerical\\n{num_c} features: Impute→Scale"]
    Pre --> Cat["📝 Categorical\\n{cat_c} features: Impute→Encode"]
    Num --> Arena["🤖 Model Arena"]
    Cat --> Arena
{model_nodes}    Arena --> Best["🏆 {best}\\n{score}% {metric}"]
{shap_node}    Best --> Export["📦 Export Bundle"]"""