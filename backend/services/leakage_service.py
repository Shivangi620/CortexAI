"""
services/leakage_service.py
Feature 4: Full data leakage and quality detector.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def run_leakage_report(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    warnings: List[str] = []
    rows = len(df)

    # ── 1. Target correlation leakage ─────────────────────────────────────────
    target_correlated: List[Dict[str, Any]] = []
    if target in df.columns:
        y = pd.to_numeric(df[target], errors="coerce")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)

        for col in num_cols:
            try:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.isna().all():
                    continue
                corr = abs(series.corr(y))
                if pd.notna(corr) and corr > 0.98:
                    target_correlated.append({
                        "column": col,
                        "correlation": round(float(corr), 4)
                    })
            except Exception:
                continue

    if target_correlated:
        names = [c["column"] for c in target_correlated]
        warnings.append(f"🔴 TARGET LEAKAGE: {names} correlates > 98% with target. Drop before training.")

    # ── 2. ID-like columns ────────────────────────────────────────────────────
    id_like: List[str] = []
    id_hints = ["id", "uuid", "uid", "idx", "row_id", "index", "key", "record"]

    for col in df.columns:
        if col == target:
            continue

        try:
            unique_ratio = df[col].nunique(dropna=True) / max(rows, 1)
        except Exception:
            unique_ratio = 0

        col_lower = str(col).lower()
        is_id_name = any(h in col_lower for h in id_hints)

        if unique_ratio > 0.9 or (is_id_name and unique_ratio > 0.5):
            id_like.append(col)

    if id_like:
        warnings.append(f"🟡 ID-LIKE COLUMNS: {id_like} have near-unique values. Likely identifiers — drop them.")

    # ── 3. Duplicate rows ──────────────────────────────────────────────────────
    try:
        dup_count = int(df.duplicated().sum())
    except Exception:
        dup_count = 0

    dup_pct = round(dup_count / max(rows, 1) * 100, 2) if rows else 0.0

    if dup_count > 0:
        warnings.append(f"🟡 DUPLICATES: {dup_count} duplicate rows ({dup_pct}%). Run df.drop_duplicates().")

    # ── 4. Constant columns ───────────────────────────────────────────────────
    constant: List[str] = []
    for col in df.columns:
        if col == target:
            continue
        try:
            if df[col].nunique(dropna=False) <= 1:
                constant.append(col)
        except Exception:
            continue

    if constant:
        warnings.append(f"🔴 CONSTANT COLUMNS: {constant} have zero variance. Drop them.")

    # ── 5. Near-constant columns ──────────────────────────────────────────────
    near_constant: List[Dict[str, Any]] = []
    for col in df.columns:
        if col == target or col in constant:
            continue
        try:
            vc = df[col].value_counts(normalize=True, dropna=True)
            if not vc.empty:
                top_freq = vc.iloc[0]
                if top_freq > 0.99:
                    near_constant.append({
                        "column": col,
                        "dominant_pct": round(float(top_freq * 100), 1)
                    })
        except Exception:
            continue

    if near_constant:
        nc_names = [c["column"] for c in near_constant]
        warnings.append(f"🟡 NEAR-CONSTANT: {nc_names} are >99% one value.")

    # ── 6. Future leakage by name ─────────────────────────────────────────────
    future_hints = ["future", "next_", "after_", "post_", "following"]
    future_leakage: List[str] = []

    for col in df.columns:
        if col == target:
            continue
        try:
            if any(h in str(col).lower() for h in future_hints):
                future_leakage.append(col)
        except Exception:
            continue

    if future_leakage:
        warnings.append(f"⚠️ TEMPORAL LEAKAGE: {future_leakage} may contain future information.")

    # ── 7. High missing ───────────────────────────────────────────────────────
    high_missing = []
    for col in df.columns:
        if col == target:
            continue
        try:
            mp = df[col].isnull().mean()
            if mp > 0.5:
                high_missing.append({
                    "column": col,
                    "missing_pct": round(float(mp * 100), 1)
                })
        except Exception:
            continue

    if high_missing:
        names = [c["column"] for c in high_missing]
        warnings.append(f"🟡 HIGH MISSING: {names} have >50% missing values.")

    # ── Summary severity ──────────────────────────────────────────────────────
    critical = len(target_correlated) > 0 or len(constant) > 0
    severity = "🔴 Critical" if critical else ("🟡 Warning" if warnings else "✅ Clean")

    if not warnings:
        warnings.append("✅ No major leakage or data quality issues detected.")

    return {
        "severity": severity,
        "target_correlated": target_correlated,
        "id_like_columns": id_like,
        "duplicate_rows": dup_count,
        "duplicate_pct": dup_pct,
        "constant_columns": constant,
        "near_constant": near_constant,
        "future_leakage": future_leakage,
        "high_missing": high_missing,
        "warnings": warnings,
        "total_issues": len(warnings),
    }