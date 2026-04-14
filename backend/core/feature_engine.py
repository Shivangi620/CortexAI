import numpy as np
import pandas as pd

# ✅ FIX 1: remove duplicate import, make optional
try:
    import featuretools as ft
except Exception:
    ft = None

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ManagedFeatureEngine:
    def __init__(self, target_col, task_type="classification", max_features=1000):
        self.target_col = target_col
        self.task_type = task_type
        self.max_features = max_features
        self.selected_features = []

    def get_dynamic_budget(self, n_rows):
        if n_rows < 1000:
            return 300
        elif n_rows < 10000:
            return 800
        return self.max_features

    def generate_features(self, df):
        """Uses Featuretools to generate automated features within budget."""

        # ✅ FIX 2: safe input
        if df is None or df.empty:
            return df

        # ✅ FIX 3: target existence
        if self.target_col not in df.columns:
            return df

        # ✅ FIX 4: featuretools optional
        if not ft:
            return df

        n_rows = len(df)
        budget = self.get_dynamic_budget(n_rows)

        # ✅ FIX 5: safe drop
        base = df.drop(columns=[self.target_col], errors="ignore").copy()

        # Normalize datetime-like columns
        for col in base.select_dtypes(include=["object"]).columns:
            sample = base[col].dropna().head(50).astype(str)
            if sample.empty:
                continue
            parsed_sample = pd.to_datetime(sample, format="mixed", errors="coerce")
            if parsed_sample.notna().mean() >= 0.8:
                base[col] = pd.to_datetime(base[col], format="mixed", errors="coerce")

        # ✅ FIX 6: wrap DFS safely
        try:
            es = ft.EntitySet(id="dataset")
            es = es.add_dataframe(
                dataframe_name="data",
                dataframe=base,
                index="id",
                make_index=True
            )

            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                max_depth=1,
                verbose=False
            )
        except Exception:
            return df

        # Add target back
        try:
            feature_matrix[self.target_col] = df[self.target_col].values
        except Exception:
            return df

        # Handle NaNs
        num_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        dt_cols = feature_matrix.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        cat_cols = feature_matrix.select_dtypes(include=["category"]).columns
        obj_cols = feature_matrix.select_dtypes(include=["object"]).columns

        if len(num_cols):
            feature_matrix[num_cols] = feature_matrix[num_cols].fillna(0)

        if len(dt_cols):
            feature_matrix[dt_cols] = feature_matrix[dt_cols].fillna(pd.Timestamp("1970-01-01"))

        for col in cat_cols:
            try:
                if "missing" not in feature_matrix[col].cat.categories:
                    feature_matrix[col] = feature_matrix[col].cat.add_categories(["missing"])
                feature_matrix[col] = feature_matrix[col].fillna("missing")
            except Exception:
                pass  # ✅ FIX 7

        if len(obj_cols):
            feature_matrix[obj_cols] = feature_matrix[obj_cols].fillna("missing")

        # Feature selection
        X = feature_matrix.drop(columns=[self.target_col], errors="ignore")
        y = feature_matrix[self.target_col]

        X_numeric = X.select_dtypes(include=['number'])

        if X_numeric.empty:
            return df

        # Mutual Info
        try:
            if self.task_type == "classification":
                mi_scores = mutual_info_classif(X_numeric, y)
            else:
                mi_scores = mutual_info_regression(X_numeric, y)
        except Exception:
            return df  # ✅ FIX 8

        mi_series = pd.Series(mi_scores, index=X_numeric.columns).sort_values(ascending=False)

        # Tree-based selection
        try:
            if self.task_type == "classification":
                selector = SelectFromModel(RandomForestClassifier(n_estimators=50, max_depth=5))
            else:
                selector = SelectFromModel(RandomForestRegressor(n_estimators=50, max_depth=5))

            selector.fit(X_numeric, y)
            tree_selected = X_numeric.columns[selector.get_support()]
        except Exception:
            tree_selected = X_numeric.columns  # ✅ FIX 9 fallback

        # Ensemble
        top_mi = mi_series.head(budget).index
        final_features = list(set(top_mi).intersection(set(tree_selected)))

        if len(final_features) < 5:
            final_features = list(mi_series.head(budget).index)

        final_features = final_features[:budget]
        self.selected_features = final_features

        # ✅ FIX 10: safe column selection
        valid_features = [f for f in final_features if f in feature_matrix.columns]

        return feature_matrix[valid_features + [self.target_col]]

    def detect_leakage(self, df):
        if df is None or df.empty:  # ✅ FIX 11
            return []

        if self.target_col not in df.columns:
            return []

        numeric_df = df.select_dtypes(include=[np.number])

        if self.target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
            return []

        try:
            correlations = numeric_df.corr()[self.target_col].abs().sort_values(ascending=False)
        except Exception:
            return []  # ✅ FIX 12

        leaks = correlations[correlations > 0.99].index.tolist()
        return [c for c in leaks if c != self.target_col]