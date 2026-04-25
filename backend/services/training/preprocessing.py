import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

try:
    from category_encoders import TargetEncoder
except Exception:
    TargetEncoder = None


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clipper to handle extreme outliers using IQR method for numerical stability."""

    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(0.25) - self.factor * (
            X.quantile(0.75) - X.quantile(0.25)
        )
        self.upper_ = X.quantile(0.75) + self.factor * (
            X.quantile(0.75) - X.quantile(0.25)
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


class SkewTransformer(BaseEstimator, TransformerMixin):
    """Auto-Log Transformer for skewed numeric features."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.skewed_cols_ = X.columns[X.skew().abs() > 0.75].tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.skewed_cols_:
            # Log transform skewed columns (ensuring non-negative)
            X[col] = np.log1p(np.maximum(X[col], 0))
        return X.values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


class SafePCA(BaseEstimator, TransformerMixin):
    """PCA wrapper that clamps n_components to the fitted matrix shape."""

    def __init__(self, n_components=None, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        n_samples, n_features = frame.shape
        max_components = min(n_samples, n_features)

        if max_components <= 1:
            self._passthrough = True
            self.n_components_ = None
            self.pca_ = None
            return self

        requested = self.n_components
        if isinstance(requested, float):
            resolved = requested
        elif requested is None:
            resolved = None
        else:
            resolved = max(1, min(int(requested), max_components))

        self._passthrough = False
        self.n_components_ = resolved
        self.pca_ = PCA(n_components=resolved, random_state=self.random_state)
        self.pca_.fit(frame, y)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        if getattr(self, "_passthrough", False) or getattr(self, "pca_", None) is None:
            return frame.values
        return self.pca_.transform(frame)

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "_passthrough", False) or getattr(self, "pca_", None) is None:
            features = input_features or []
            return np.asarray(features, dtype=object)

        width = int(getattr(self.pca_, "n_components_", 0) or 0)
        return np.asarray([f"pca_{idx}" for idx in range(width)], dtype=object)


def fuzzy_merge_labels(series: pd.Series, threshold=0.9):
    """Merge similar categorical labels (e.g., 'Male' vs 'male' or 'Heart' vs 'Heartz')."""
    counts = series.value_counts()
    if len(counts) > 50:
        return series  # Too many unique labels for fuzzy

    unique_labels = counts.index.tolist()
    mapping = {}

    for i, label1 in enumerate(unique_labels):
        if label1 in mapping:
            continue
        mapping[label1] = label1
        for label2 in unique_labels[i + 1 :]:
            if label2 in mapping:
                continue
            # Similarity ignoring case
            if (
                SequenceMatcher(None, label1.lower(), label2.lower()).ratio()
                > threshold
            ):
                # Merge the less frequent label into the more frequent one
                mapping[label2] = label1

    return series.map(mapping)


def smart_extract_numeric(series: pd.Series):
    """Scan a categorical column for hidden numeric patterns (e.g., '120/80', '10kg')."""
    sample = series.dropna().head(20).astype(str)

    # Check for unit patterns ($100, 100%, 10kg)
    unit_match = sample.str.contains(
        r"^\s*[\$\€\£]?\s*[\d\.\,]+\s*[\w\%]*\s*$", regex=True
    ).mean()
    if unit_match > 0.7:
        # Extract single number
        extracted = (
            series.astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float)
        )
        return {series.name: extracted}

    # Check for pair patterns (120/80, 5-10)
    pair_match = sample.str.contains(
        r"(?:\d+)\/(?:\d+)|(?:\d+)-(?:\d+)", regex=True
    ).mean()
    if pair_match > 0.5:
        # Extract pairs
        extracted = series.astype(str).str.extract(r"(\d+)[/-](\d+)", expand=True)
        if not extracted.isnull().all().all():
            return {
                f"{series.name}_part1": extracted[0].astype(float),
                f"{series.name}_part2": extracted[1].astype(float),
            }

    return None


def auto_clean_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Automatically scrub the dataset for common quality issues in any domain."""
    logs = []
    clean_df = df.copy()
    initial_rows = len(clean_df)
    original_columns = list(clean_df.columns)

    clean_df.columns = [str(col).strip().replace("\n", " ") for col in clean_df.columns]
    if clean_df.columns.tolist() != original_columns:
        logs.append("Normalized column names by trimming whitespace.")

    # ── 1. Universal Null Masking ───────────────────────────────────────────
    null_placeholders = [
        "??",
        "nan",
        "none",
        "null",
        "unknown",
        "na",
        "n/a",
        "invalid",
        "?",
        "-",
        "none",
    ]
    for col in clean_df.columns:
        if clean_df[col].dtype == "object":
            clean_df[col] = clean_df[col].replace(null_placeholders, np.nan)
            # Standardize case/space
            clean_df[col] = clean_df[col].astype(str).str.strip().replace("nan", np.nan)
            lower_vals = clean_df[col].dropna().astype(str).str.lower()
            if (
                not lower_vals.empty
                and lower_vals.isin(["true", "false", "yes", "no", "0", "1"]).mean()
                > 0.8
            ):
                clean_df[col] = lower_vals.map(
                    {
                        "true": True,
                        "false": False,
                        "yes": True,
                        "no": False,
                        "1": True,
                        "0": False,
                    }
                )
                logs.append(f"Normalized boolean-like values in '{col}'.")

    # ── 2. Deduplication ────────────────────────────────────────────────────
    clean_df.drop_duplicates(inplace=True)
    if len(clean_df) < initial_rows:
        logs.append(f"Removed {initial_rows - len(clean_df)} duplicate rows.")

    # ── 3. Smart Extraction & Label Merging ────────────────────────────────
    cols_to_drop = []
    id_hints = ["id", "uuid", "uid", "idx", "row_id", "timestamp", "created_at"]

    for col in clean_df.columns:
        if col == target:
            continue

        # A. High Missing Values (>90%)
        missing_pct = clean_df[col].isnull().mean()
        if missing_pct > 0.90:
            cols_to_drop.append(col)
            logs.append(f"Dropped '{col}' (>90% missing).")
            continue

        # B. Low Variance / Constant Values
        if clean_df[col].nunique() <= 1:
            cols_to_drop.append(col)
            logs.append(f"Dropped '{col}' (zero variance).")
            continue

        # C. Clear Identifiers
        if (
            any(h in col.lower() for h in id_hints)
            and clean_df[col].nunique() > len(clean_df) * 0.8
        ):
            cols_to_drop.append(col)
            logs.append(f"Dropped '{col}' (probable identifier).")
            continue

        # D. Smart Numeric Extraction
        if clean_df[col].dtype == "object":
            extracted_features = smart_extract_numeric(clean_df[col])
            if extracted_features:
                for new_col, data in extracted_features.items():
                    clean_df[new_col] = data
                cols_to_drop.append(col)
                logs.append(f"Extracted numeric features from '{col}'.")
                continue

            # E. Fuzzy Category Consolidation (Merge typos)
            if clean_df[col].nunique() < 30:
                original_n = clean_df[col].nunique()
                clean_df[col] = fuzzy_merge_labels(clean_df[col])
                if clean_df[col].nunique() < original_n:
                    logs.append(
                        f"Consolidated similar labels in '{col}' (merged {original_n - clean_df[col].nunique()} variants)."
                    )

    if cols_to_drop:
        clean_df.drop(columns=cols_to_drop, inplace=True)

    return clean_df, logs


def extract_datetime_features(df):
    """Automatically extract datetime component features from object columns."""
    new_df = df.copy()
    for col in new_df.columns:
        if new_df[col].dtype == "object":
            sample = new_df[col].dropna().head(10)
            if sample.empty:
                continue
            try:
                pd.to_datetime(sample, format="mixed", errors="raise")
                new_df[col] = pd.to_datetime(new_df[col], errors="coerce")
                new_df[f"{col}_year"] = new_df[col].dt.year
                new_df[f"{col}_month"] = new_df[col].dt.month
                new_df[f"{col}_day"] = new_df[col].dt.day
                new_df[f"{col}_is_weekend"] = new_df[col].dt.weekday >= 5
                new_df.drop(columns=[col], inplace=True)
            except Exception:
                pass
    return new_df


def _resolve_pca_components(num_cols, pca_mode: str = "auto", pca_components: int = 0):
    mode = str(pca_mode or "auto").strip().lower()
    requested = int(pca_components or 0)
    numeric_count = len(num_cols or [])

    if numeric_count <= 1:
        return None
    if mode == "off":
        return None
    if mode == "always":
        if requested > 1:
            return min(requested, numeric_count - 1)
        return min(24, max(8, numeric_count // 3))
    if numeric_count >= 40:
        return min(24, max(8, numeric_count // 3))
    return None


def make_preprocessor(num_cols, cat_cols, pca_mode: str = "auto", pca_components: int = 0):
    """Factory: returns a fresh, unfitted ColumnTransformer per call."""
    transformers = []

    # Numeric transformer
    if num_cols:
        use_interactions = len(num_cols) <= 18
        resolved_pca_components = _resolve_pca_components(
            num_cols, pca_mode=pca_mode, pca_components=pca_components
        )

        num_steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("skew_fix", SkewTransformer()),
            ("outlier_clipper", OutlierClipper(factor=3.0)),
        ]
        if use_interactions:
            num_steps.append(
                (
                    "interactions",
                    PolynomialFeatures(
                        degree=2, interaction_only=True, include_bias=False
                    ),
                )
            )
        num_steps.append(("scaler", StandardScaler()))
        if resolved_pca_components:
            num_steps.append(
                ("pca", SafePCA(n_components=resolved_pca_components, random_state=42))
            )

        num_transformer = Pipeline(steps=num_steps)
        transformers.append(("num", num_transformer, num_cols))

    # Categorical transformer
    if cat_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ]
        if TargetEncoder is not None:
            cat_steps.append(
                ("target_encoder", TargetEncoder(smoothing=0.3, min_samples_leaf=10))
            )
        else:
            cat_steps.append(
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )
        cat_transformer = Pipeline(steps=cat_steps)
        transformers.append(("cat", cat_transformer, cat_cols))

    if not transformers:
        raise ValueError(
            "Neither numeric nor categorical columns found. Cannot create preprocessor."
        )

    return ColumnTransformer(transformers=transformers)


def make_lite_preprocessor(num_cols, cat_cols, pca_mode: str = "auto", pca_components: int = 0):
    """Featherweight preprocessor for Stage 1 sweeps."""
    transformers = []

    # Numeric transformer
    if num_cols:
        num_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        resolved_pca_components = _resolve_pca_components(
            num_cols, pca_mode=pca_mode, pca_components=pca_components
        )
        if resolved_pca_components:
            num_steps.append(
                (
                    "pca",
                    SafePCA(
                        n_components=min(resolved_pca_components, min(16, max(6, len(num_cols) // 8))),
                        random_state=42,
                    ),
                )
            )
        num_transformer = Pipeline(steps=num_steps)
        transformers.append(("num", num_transformer, num_cols))

    # Categorical transformer
    if cat_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ]
        if TargetEncoder is not None:
            cat_steps.append(("target_encoder", TargetEncoder()))
        else:
            cat_steps.append(
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            )
        cat_transformer = Pipeline(steps=cat_steps)
        transformers.append(("cat", cat_transformer, cat_cols))

    if not transformers:
        raise ValueError(
            "Neither numeric nor categorical columns found. Cannot create preprocessor."
        )

    return ColumnTransformer(transformers=transformers)


class DataAgent:
    """Specialized Agent for dataset DNA analysis and cleaning."""

    def __init__(self):
        self.reasoning = []

    def clean(self, df: pd.DataFrame, target: str):
        self.reasoning.append("DataAgent: Initiating deep cleaning protocol.")
        df, logs = auto_clean_data(df, target)
        for log in logs:
            self.reasoning.append(f"DataAgent Decision: {log}")
        df = extract_datetime_features(df)
        self.reasoning.append(
            "DataAgent: Completed automated feature engineering (Datetime)."
        )
        return df, self.reasoning
