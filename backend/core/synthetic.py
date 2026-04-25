import pandas as pd
import numpy as np


def generate_synthetic(
    df: pd.DataFrame, n_new_rows: int, random_state: int = 42
) -> tuple:
    """
    Generate synthetic rows while preserving the original dataset's type contract.

    The generator uses a seed-row approach so row-level relationships survive better
    than simple per-column independent sampling, and it restores column dtypes after
    generation instead of coercing the entire frame toward numeric values.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be empty")
    if n_new_rows <= 0:
        raise ValueError("n_new_rows must be positive")

    df_clean = df.copy()
    rng = np.random.default_rng(random_state)
    profiles = {column: _build_column_profile(df_clean[column]) for column in df_clean.columns}
    non_empty = df_clean.dropna(how="all")
    seed_pool = non_empty if not non_empty.empty else df_clean

    synthetic_rows = []
    existing_signatures = {_row_signature(row) for _, row in df_clean.iterrows()}

    max_attempts = max(n_new_rows * 5, 25)
    attempts = 0
    while len(synthetic_rows) < n_new_rows and attempts < max_attempts:
        attempts += 1
        seed_row = seed_pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        row = {}

        for column in df_clean.columns:
            profile = profiles[column]
            if rng.random() < profile["missing_ratio"]:
                row[column] = _missing_value_for_profile(profile)
                continue
            row[column] = _sample_value(column, seed_row.get(column), profile, rng)

        signature = _row_signature(pd.Series(row))
        if signature in existing_signatures:
            row = _nudge_duplicate_row(row, profiles, rng)
            signature = _row_signature(pd.Series(row))

        synthetic_rows.append(row)
        existing_signatures.add(signature)

    while len(synthetic_rows) < n_new_rows:
        seed_row = seed_pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        row = {}
        for column in df_clean.columns:
            profile = profiles[column]
            if rng.random() < profile["missing_ratio"]:
                row[column] = _missing_value_for_profile(profile)
                continue
            row[column] = _sample_value(column, seed_row.get(column), profile, rng)
        row = _nudge_duplicate_row(row, profiles, rng)
        synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows, columns=df_clean.columns)
    synthetic_df = _restore_dataframe_types(synthetic_df, df_clean, profiles)
    expanded = pd.concat([df_clean, synthetic_df], ignore_index=True)
    return expanded, synthetic_df


def _build_column_profile(series: pd.Series) -> dict:
    missing_ratio = float(series.isna().mean()) if len(series) else 0.0

    bool_like_object = False
    if series.dtype == object:
        non_null = series.dropna()
        if not non_null.empty:
            normalized = non_null.map(_normalize_bool_like_value)
            bool_like_object = normalized.notna().all()

    if pd.api.types.is_bool_dtype(series):
        values = series.dropna().astype(bool)
        freq = values.value_counts(normalize=True, dropna=True)
        return {
            "kind": "bool",
            "missing_ratio": missing_ratio,
            "values": freq.index.tolist(),
            "probabilities": freq.values.tolist(),
            "dtype": str(series.dtype),
        }

    if bool_like_object:
        normalized = series.dropna().map(_normalize_bool_like_value).astype(bool)
        freq = normalized.value_counts(normalize=True, dropna=True)
        return {
            "kind": "bool",
            "missing_ratio": missing_ratio,
            "values": freq.index.tolist(),
            "probabilities": freq.values.tolist(),
            "dtype": str(series.dtype),
            "force_nullable_boolean": True,
        }

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        values = parsed.dropna().sort_values()
        return {
            "kind": "datetime",
            "missing_ratio": missing_ratio,
            "values": values,
            "min": values.min() if not values.empty else None,
            "max": values.max() if not values.empty else None,
            "dtype": str(series.dtype),
        }

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        quantiles = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if not valid.empty else pd.Series(dtype=float)
        integer_like = bool(
            not valid.empty and np.allclose(valid, np.round(valid), equal_nan=True)
        )
        return {
            "kind": "numeric",
            "missing_ratio": missing_ratio,
            "values": valid,
            "min": float(valid.min()) if not valid.empty else 0.0,
            "max": float(valid.max()) if not valid.empty else 0.0,
            "std": float(valid.std()) if len(valid) > 1 else 0.0,
            "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]) if len(quantiles) else 0.0,
            "median": float(quantiles.loc[0.5]) if len(quantiles) else 0.0,
            "integer_like": integer_like,
            "dtype": str(series.dtype),
        }

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if not series.dropna().empty and float(parsed.notna().mean()) >= 0.8:
        values = parsed.dropna().sort_values()
        return {
            "kind": "datetime",
            "missing_ratio": missing_ratio,
            "values": values,
            "min": values.min() if not values.empty else None,
            "max": values.max() if not values.empty else None,
            "dtype": str(series.dtype),
        }

    text = series.dropna().astype(str)
    freq = text.value_counts(normalize=True, dropna=True)
    return {
        "kind": "categorical",
        "missing_ratio": missing_ratio,
        "values": freq.index.tolist(),
        "probabilities": freq.values.tolist(),
        "dtype": str(series.dtype),
        "high_cardinality": len(freq) > max(20, len(series) * 0.25),
    }


def _sample_value(column: str, seed_value, profile: dict, rng: np.random.Generator):
    kind = profile["kind"]
    if kind == "numeric":
        return _sample_numeric(seed_value, profile, rng)
    if kind == "bool":
        return _sample_bool(seed_value, profile, rng)
    if kind == "datetime":
        return _sample_datetime(seed_value, profile, rng)
    return _sample_categorical(seed_value, profile, rng)


def _sample_numeric(seed_value, profile: dict, rng: np.random.Generator):
    values = profile["values"]
    if values.empty:
        return np.nan

    seed_numeric = pd.to_numeric(pd.Series([seed_value]), errors="coerce").iloc[0]
    if pd.isna(seed_numeric):
        seed_numeric = float(values.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0])

    spread = profile["std"] or (profile["iqr"] / 1.35) or max(abs(profile["median"]) * 0.05, 1.0)
    noise = rng.normal(0.0, max(spread * 0.12, 1e-9))
    candidate = float(seed_numeric + noise)
    candidate = float(np.clip(candidate, profile["min"], profile["max"]))

    if profile["integer_like"]:
        return int(round(candidate))
    return round(candidate, min(_infer_decimals(values), 6))


def _sample_bool(seed_value, profile: dict, rng: np.random.Generator):
    if seed_value is not None and not pd.isna(seed_value) and rng.random() < 0.7:
        return bool(seed_value)
    values = profile["values"] or [True, False]
    probabilities = profile["probabilities"] or [0.5, 0.5]
    return bool(rng.choice(values, p=probabilities))


def _sample_datetime(seed_value, profile: dict, rng: np.random.Generator):
    values = profile["values"]
    if values.empty:
        return pd.NaT

    seed_ts = pd.to_datetime(seed_value, errors="coerce")
    if pd.isna(seed_ts):
        seed_ts = values.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]

    if len(values) > 1:
        diffs = values.sort_values().diff().dropna()
        jitter_scale = max(int(diffs.median().total_seconds() if not diffs.empty else 0), 0)
    else:
        jitter_scale = 0

    if jitter_scale > 0:
        offset_seconds = int(rng.normal(0, max(jitter_scale * 0.2, 1)))
        candidate = seed_ts + pd.to_timedelta(offset_seconds, unit="s")
    else:
        candidate = seed_ts

    if profile["min"] is not None and candidate < profile["min"]:
        candidate = profile["min"]
    if profile["max"] is not None and candidate > profile["max"]:
        candidate = profile["max"]
    return candidate


def _sample_categorical(seed_value, profile: dict, rng: np.random.Generator):
    values = profile["values"]
    probabilities = profile["probabilities"]
    if seed_value is not None and not pd.isna(seed_value) and rng.random() < (0.8 if profile["high_cardinality"] else 0.65):
        return str(seed_value)
    if not values:
        return None
    return rng.choice(values, p=probabilities)


def _restore_dataframe_types(df: pd.DataFrame, source_df: pd.DataFrame, profiles: dict) -> pd.DataFrame:
    restored = df.copy()
    for column in restored.columns:
        profile = profiles[column]
        kind = profile["kind"]
        source_dtype = source_df[column].dtype

        if kind == "numeric":
            restored[column] = pd.to_numeric(restored[column], errors="coerce")
            if profile["integer_like"] and restored[column].notna().any():
                try:
                    restored[column] = restored[column].round().astype(source_dtype)
                except Exception:
                    restored[column] = restored[column].round().astype("Int64")
        elif kind == "bool":
            normalized = restored[column].map(_normalize_bool_like_value)
            if profile.get("force_nullable_boolean") or str(source_dtype) == "object":
                restored[column] = normalized.astype("boolean")
            else:
                try:
                    restored[column] = normalized.astype(source_dtype)
                except Exception:
                    restored[column] = normalized.astype("boolean")
        elif kind == "datetime":
            restored[column] = pd.to_datetime(restored[column], errors="coerce", format="mixed")
        else:
            restored[column] = restored[column].astype(object)
            if str(source_dtype) == "category":
                categories = source_df[column].astype("category").cat.categories
                restored[column] = pd.Categorical(restored[column], categories=categories)
    return restored


def _nudge_duplicate_row(row: dict, profiles: dict, rng: np.random.Generator) -> dict:
    updated = dict(row)
    numeric_columns = [column for column, profile in profiles.items() if profile["kind"] == "numeric" and not profile["values"].empty]
    if not numeric_columns:
        return updated
    column = rng.choice(numeric_columns)
    updated[column] = _sample_numeric(updated.get(column), profiles[column], rng)
    return updated


def _missing_value_for_profile(profile: dict):
    if profile["kind"] == "datetime":
        return pd.NaT
    return np.nan


def _row_signature(row: pd.Series) -> tuple:
    return tuple("<MISSING>" if pd.isna(value) else str(value) for value in row.tolist())


def _infer_decimals(series: pd.Series) -> int:
    sample = series.dropna().head(20).astype(str)
    decimals = []
    for value in sample:
        if "." in value:
            decimals.append(len(value.split(".")[-1]))
    return max(decimals) if decimals else 0


def _normalize_bool_like_value(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return pd.NA


def suggest_expansion_size(rows: int) -> int:
    if rows < 100:
        return max(100, rows * 10)
    if rows < 500:
        return rows * 4
    if rows < 1000:
        return rows * 2
    return rows // 2
