import pandas as pd
import numpy as np


def generate_synthetic(df: pd.DataFrame, n_new_rows: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    synthetic_rows = []

    for _ in range(n_new_rows):
        row = {}

        # Numerical
        if num_cols:
            sampled = df[num_cols].dropna()
            if sampled.empty:
                continue

            seed_row = sampled.sample(
                1, random_state=int(rng.integers(0, 1_000_000))
            ).iloc[0]

            for col in num_cols:
                col_std = df[col].std()
                if pd.isna(col_std):
                    col_std = 0

                noise = rng.normal(0, col_std * 0.15)
                val = seed_row[col] + noise

                try:
                    val = float(np.clip(val, df[col].min(), df[col].max()))
                except Exception:
                    val = seed_row[col]

                decimals = _infer_decimals(df[col])
                row[col] = round(val, decimals)

        # Categorical
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True, dropna=True)
            if freq.empty:
                row[col] = None
                continue
            row[col] = rng.choice(freq.index.tolist(), p=freq.values.tolist())

        synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows, columns=df.columns)

    # CLEAN TYPES
    synthetic_df = synthetic_df.replace("", np.nan)

    for col in synthetic_df.columns:
        try:
            synthetic_df[col] = pd.to_numeric(synthetic_df[col])
        except Exception:
            pass

    # FIX: clean BOTH original + synthetic
    df = df.replace("", np.nan)
    synthetic_df = synthetic_df.replace("", np.nan)

    for col in synthetic_df.columns:
        try:
            synthetic_df[col] = pd.to_numeric(synthetic_df[col])
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    expanded = pd.concat([df, synthetic_df], ignore_index=True)
    return expanded, synthetic_df


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