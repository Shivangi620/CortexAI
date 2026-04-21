import pandas as pd
import numpy as np


def generate_synthetic(
    df: pd.DataFrame, n_new_rows: int, random_state: int = 42
) -> tuple:
    """
    Generate synthetic rows based on the input dataframe statistics.

    Args:
        df: Input dataframe to base generation on
        n_new_rows: Number of synthetic rows to generate
        random_state: Random seed for reproducibility

    Returns:
        tuple: (expanded_df with original + synthetic, synthetic_df with only new rows)
    """
    # Input validation
    if df is None or df.empty:
        raise ValueError("Input dataframe cannot be empty")
    if n_new_rows <= 0:
        raise ValueError("n_new_rows must be positive")

    # Make a copy to avoid modifying the input
    df_clean = df.copy()
    rng = np.random.default_rng(random_state)

    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()

    synthetic_rows = []
    rows_generated = 0
    max_attempts = n_new_rows * 3  # Allow multiple attempts in case of NaN skipping

    for attempt in range(max_attempts):
        if rows_generated >= n_new_rows:
            break

        row = {}

        # Numerical columns
        if num_cols:
            sampled = df_clean[num_cols].dropna()
            if sampled.empty:
                # All NaN values - use column statistics instead
                for col in num_cols:
                    col_mean = df_clean[col].mean()
                    col_std = df_clean[col].std()
                    if pd.isna(col_mean):
                        col_mean = 0
                    if pd.isna(col_std) or col_std == 0:
                        col_std = 1
                    noise = rng.normal(0, col_std * 0.15)
                    val = float(
                        np.clip(
                            col_mean + noise, df_clean[col].min(), df_clean[col].max()
                        )
                    )
                    decimals = _infer_decimals(df_clean[col])
                    row[col] = round(val, decimals)
            else:
                seed_row = sampled.sample(
                    1, random_state=int(rng.integers(0, 1_000_000))
                ).iloc[0]

                for col in num_cols:
                    col_std = df_clean[col].std()
                    if pd.isna(col_std):
                        col_std = 0

                    noise = rng.normal(0, col_std * 0.15)
                    val = seed_row[col] + noise

                    try:
                        val = float(
                            np.clip(val, df_clean[col].min(), df_clean[col].max())
                        )
                    except (ValueError, TypeError):
                        val = seed_row[col]

                    decimals = _infer_decimals(df_clean[col])
                    row[col] = round(val, decimals)

        # Categorical columns
        for col in cat_cols:
            freq = df_clean[col].value_counts(normalize=True, dropna=True)
            if freq.empty:
                row[col] = None
            else:
                try:
                    row[col] = rng.choice(freq.index.tolist(), p=freq.values.tolist())
                except (ValueError, IndexError):
                    row[col] = None

        synthetic_rows.append(row)
        rows_generated += 1

    synthetic_df = pd.DataFrame(synthetic_rows, columns=df_clean.columns)

    # Type conversion and cleaning (single pass)
    synthetic_df = _clean_dataframe_types(synthetic_df)

    # Concatenate and return
    expanded = pd.concat([df_clean, synthetic_df], ignore_index=True)
    return expanded, synthetic_df


def _clean_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe by replacing empty strings with NaN and converting numeric columns.

    Args:
        df: Dataframe to clean

    Returns:
        Cleaned dataframe
    """
    df_copy = df.copy()
    df_copy = df_copy.replace("", np.nan)

    for col in df_copy.columns:
        try:
            # Try to convert to numeric, coerce errors to NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        except Exception:
            # If conversion fails, keep original type
            pass

    return df_copy


def _infer_decimals(series: pd.Series) -> int:
    sample = series.dropna().head(20).astype(str)
    decimals = []
    for v in sample:
        if "." in v:
            decimals.append(len(v.split(".")[-1]))
    return max(decimals) if decimals else 0


def suggest_expansion_size(rows: int) -> int:
    if rows < 100:
        return max(100, rows * 10)
    elif rows < 500:
        return rows * 4
    elif rows < 1000:
        return rows * 2
    else:
        return rows // 2
