import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame) -> dict:
    # ✅ FIX 1: handle empty dataframe
    if df is None or df.empty:
        return {"error": "Empty dataset"}

    rows, cols = df.shape

    if rows < 1000:
        size = "Small"
    elif rows < 100000:
        size = "Medium"
    else:
        size = "Large"

    # Missing values
    total_cells = rows * cols
    missing_by_col = df.isnull().sum()
    missing_cells = int(missing_by_col.sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0

    # Sampling
    is_sampled = rows > 100000
    if is_sampled:
        # ✅ FIX 2: prevent crash if rows < 100000 due to edge conditions
        sample_n = min(100000, rows)
        df_stats = df.sample(n=sample_n, random_state=42)
    else:
        df_stats = df

    columns = df.columns.tolist()

    target_names = [
        'target', 'label', 'class', 'outcome', 'result', 'churn', 'price',
        'default', 'y', 'survived', 'fraud', 'output', 'pred', 'prediction',
        'target_class', 'target_label'
    ]
    id_hints = ['id', 'uuid', 'uid', 'index', 'idx', 'timestamp', 'date', 'time', 'row_id']

    suggested_target = None

    for col in columns:
        # ✅ FIX 3: safe string conversion
        if str(col).lower() in target_names:
            suggested_target = col
            break

    if not suggested_target and columns:
        for col in reversed(columns):
            if not any(h in str(col).lower() for h in id_hints):
                suggested_target = col
                break

        if not suggested_target:
            suggested_target = columns[-1]

    # Feature types
    num_cols = df_stats.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_stats.select_dtypes(include=['object', 'category']).columns.tolist()

    # Model suggestion
    suggested_model = "Tree-based (Random Forest / XGBoost)"
    if rows > 10000 and len(num_cols) > len(cat_cols):
        suggested_model = "Gradient Boosting (XGBoost/LightGBM)"
    elif len(num_cols) > 0 and len(cat_cols) == 0 and rows < 5000:
        suggested_model = "Linear Model / SVM"

    # Imbalance
    imbalance = "Low"
    if suggested_target and suggested_target in df_stats.columns:
        target_counts = df_stats[suggested_target].value_counts()

        # ✅ FIX 4: avoid division errors
        if len(target_counts) >= 2 and target_counts.iloc[1] != 0:
            ratio = target_counts.iloc[0] / target_counts.iloc[1]
            if ratio > 3 or ratio < 0.33:
                imbalance = "High ⚠️"

    column_stats = {}
    feature_types = {}

    for col in columns:
        # ✅ FIX 5: safe column access
        if col not in df_stats.columns:
            continue

        col_series_stats = df_stats[col]

        unique_count = int(col_series_stats.nunique())
        unique_pct = unique_count / (len(df_stats) if len(df_stats) > 0 else 1)

        stats = {
            "dtype": str(df[col].dtype),
            "missing": int(missing_by_col.get(col, 0)),  # ✅ FIX 6
            "missing_pct": round(float(missing_by_col.get(col, 0) / rows * 100), 1) if rows > 0 else 0,
            "unique": unique_count,
            "unique_pct": unique_pct,
            "outliers": 0
        }

        # Semantic typing
        semantic_type = "Unknown"

        if unique_count == 2:
            semantic_type = "Binary"

        elif any(id_str in str(col).lower() for id_str in ['id', 'uuid', 'index']) and unique_pct > 0.8:
            semantic_type = "ID/Index"

        elif col in num_cols:
            # ✅ FIX 7: robust dtype check
            if pd.api.types.is_float_dtype(df[col]):
                semantic_type = "Continuous"
            elif unique_count < 20:
                semantic_type = "Discrete/Ordinal"
            else:
                semantic_type = "Continuous"

        else:
            # ✅ FIX 8: proper datetime detection
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                semantic_type = "DateTime"
            else:
                semantic_type = "Nominal Category"

        feature_types[col] = semantic_type
        stats["semantic_type"] = semantic_type

        # Numeric stats
        if col in num_cols and not col_series_stats.isnull().all():
            try:
                stats["mean"] = round(float(col_series_stats.mean()), 4)
                stats["std"] = round(float(col_series_stats.std()), 4)
                stats["min"] = round(float(col_series_stats.min()), 4)
                stats["max"] = round(float(col_series_stats.max()), 4)

                stats["skew"] = (
                    round(float(col_series_stats.skew()), 1)
                    if not is_sampled else "N/A (Sampled)"
                )

                # Outliers (IQR)
                q1 = col_series_stats.quantile(0.25)
                q3 = col_series_stats.quantile(0.75)
                iqr = q3 - q1

                if iqr != 0:  # ✅ FIX 9: avoid zero division
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = col_series_stats[
                        (col_series_stats < lower_bound) | (col_series_stats > upper_bound)
                    ]

                    stats["outliers"] = int(len(outliers))
                    stats["outlier_pct"] = round(
                        (stats["outliers"] / len(df_stats)) * 100, 2
                    )

            except Exception:
                pass  # ✅ FIX 10: prevent crash on bad data

        elif col in cat_cols:
            try:
                top_vals = col_series_stats.value_counts().head(3)
                stats["top_values"] = top_vals.index.tolist()
            except Exception:
                stats["top_values"] = []

        column_stats[col] = stats

    task_type = "classification"
    if suggested_target and suggested_target in df.columns:
        target_series = df[suggested_target].dropna()
        if not target_series.empty:
            if pd.api.types.is_numeric_dtype(target_series):
                unique_count = target_series.nunique(dropna=True)
                unique_ratio = unique_count / max(len(target_series), 1)
                if pd.api.types.is_float_dtype(target_series) or not (
                    unique_count <= 20 and unique_ratio <= 0.2
                ):
                    task_type = "regression"

    # Health score
    try:
        from core.health_score import compute_health_score

        health_metadata = compute_health_score({
            "rows": rows,
            "cols": cols,
            "missing_pct": missing_pct,
            "imbalance": imbalance,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "column_stats": column_stats
        })
    except Exception:
        health_metadata = {"error": "health_score unavailable"}  # ✅ FIX 11

    return {
        "rows": rows,
        "cols": cols,
        "size": size,
        "missing_pct": missing_pct,
        "missing_values": missing_cells,
        "columns": columns,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "imbalance": imbalance,
        "suggested_target": suggested_target,
        "task_type": task_type,
        "suggested_model": suggested_model,
        "column_stats": column_stats,
        "is_sampled": is_sampled,
        "sample_size": len(df_stats),  # ✅ FIX 12 (accurate)
        "health": health_metadata
    }
