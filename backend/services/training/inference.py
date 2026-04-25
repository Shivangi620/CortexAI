from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

from services.data_sanitizer import sanitize_dataframe
from services.training.preprocessing import make_lite_preprocessor, make_preprocessor


class DeterministicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Reusable raw-tabular cleaner/feature builder that is safe at inference time."""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns

    def fit(self, X, y=None):
        frame = self._coerce_frame(X)
        self.feature_columns_ = (
            list(self.feature_columns) if self.feature_columns else list(frame.columns)
        )
        frame = frame.reindex(columns=self.feature_columns_, fill_value=np.nan)
        sanitized = sanitize_dataframe(frame, drop_duplicate_rows=False)
        clean = sanitized.df.reindex(columns=self.feature_columns_, fill_value=np.nan)

        datetime_columns: List[str] = []
        for column in clean.columns:
            series = clean[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(column)
                continue
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue
            sample = series.dropna().astype(str).head(100)
            if sample.empty:
                continue
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if float(parsed.notna().mean()) >= 0.8:
                datetime_columns.append(column)

        self.datetime_columns_ = sorted(set(datetime_columns))
        transformed = self._transform_frame(clean)
        self.output_columns_ = list(transformed.columns)
        return self

    def transform(self, X):
        frame = self._coerce_frame(X)
        expected = getattr(self, "feature_columns_", self.feature_columns)
        frame = frame.reindex(columns=expected, fill_value=np.nan)
        sanitized = sanitize_dataframe(frame, drop_duplicate_rows=False)
        clean = sanitized.df.reindex(columns=expected, fill_value=np.nan)
        transformed = self._transform_frame(clean)
        output_columns = getattr(self, "output_columns_", list(transformed.columns))
        return transformed.reindex(columns=output_columns, fill_value=np.nan)

    def get_feature_names_out(self):
        return np.asarray(getattr(self, "output_columns_", []), dtype=object)

    def _coerce_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame([X])
        return pd.DataFrame(X)

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in getattr(self, "datetime_columns_", []):
            if column not in out.columns:
                continue
            parsed = pd.to_datetime(out[column], errors="coerce", format="mixed")
            out[f"{column}_year"] = parsed.dt.year
            out[f"{column}_month"] = parsed.dt.month
            out[f"{column}_day"] = parsed.dt.day
            out[f"{column}_is_weekend"] = parsed.dt.weekday >= 5
            out = out.drop(columns=[column])
        return out


class TabularModelPipeline(BaseEstimator):
    """End-to-end model that accepts raw tabular input."""

    def __init__(
        self,
        base_estimator,
        feature_columns: Optional[List[str]] = None,
        preprocessing: str = "lite",
        pca_mode: str = "auto",
        pca_components: int = 0,
    ):
        self.base_estimator = base_estimator
        self.feature_columns = feature_columns
        self.preprocessing = preprocessing
        self.pca_mode = pca_mode
        self.pca_components = pca_components

    def fit(self, X, y):
        self.feature_engineer_ = DeterministicFeatureEngineer(
            feature_columns=self.feature_columns
        )
        X_features = self.feature_engineer_.fit_transform(X, y)

        self.num_cols_ = X_features.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X_features.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        X_features = self._normalize_model_frame(X_features, self.cat_cols_)

        if self.preprocessing == "full":
            self.preprocessor_ = make_preprocessor(
                self.num_cols_,
                self.cat_cols_,
                pca_mode=self.pca_mode,
                pca_components=self.pca_components,
            )
        else:
            self.preprocessor_ = make_lite_preprocessor(
                self.num_cols_,
                self.cat_cols_,
                pca_mode=self.pca_mode,
                pca_components=self.pca_components,
            )

        X_prepared = self.preprocessor_.fit_transform(X_features, y)
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_prepared, y)
        self.feature_names_in_ = list(getattr(self.feature_engineer_, "feature_columns_", []))
        try:
            self.feature_names_out_ = list(self.preprocessor_.get_feature_names_out())
        except Exception:
            self.feature_names_out_ = [
                f"feature_{idx}" for idx in range(getattr(X_prepared, "shape", [0, 0])[1])
            ]
        return self

    def predict(self, X):
        X_prepared = self.preprocess(X)
        return self.model_.predict(X_prepared)

    def predict_proba(self, X):
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError("Underlying estimator does not support predict_proba")
        X_prepared = self.preprocess(X)
        return self.model_.predict_proba(X_prepared)

    def preprocess(self, X):
        X_features = self.transform_features(X)
        X_features = self._normalize_model_frame(
            X_features, getattr(self, "cat_cols_", [])
        )
        return self.preprocessor_.transform(X_features)

    def transform_features(self, X) -> pd.DataFrame:
        return self.feature_engineer_.transform(X)

    def _normalize_model_frame(self, frame: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        out = frame.copy()
        for column in cat_cols or []:
            if column in out.columns:
                out[column] = out[column].astype(object)
        return out

    def get_feature_names_out(self):
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)

    @property
    def classes_(self):
        return getattr(self.model_, "classes_", None)


class InferenceArtifact:
    """Pickle-friendly inference bundle with label decoding."""

    def __init__(
        self,
        model: TabularModelPipeline,
        label_encoder=None,
        task_type: str = "regression",
        target_name: str = "",
        raw_feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.label_encoder = label_encoder
        self.task_type = task_type
        self.target_name = target_name
        self.feature_names_in_ = list(
            raw_feature_names
            or getattr(model, "feature_names_in_", [])
        )
        self.feature_names_out_ = list(getattr(model, "feature_names_out_", []))
        self.classes_ = self._decoded_classes()

    def predict(self, X):
        encoded = self.model.predict(X)
        return self._decode_predictions(encoded)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def preprocess(self, X):
        return self.model.preprocess(X)

    def transform_features(self, X):
        return self.model.transform_features(X)

    def get_feature_names_out(self):
        return np.asarray(self.feature_names_out_, dtype=object)

    def get_underlying_model(self):
        return self.model.model_

    def predict_encoded(self, X):
        return self.model.predict(X)

    def _decode_predictions(self, values):
        if self.label_encoder is None:
            return values
        arr = np.asarray(values)
        try:
            decoded = self.label_encoder.inverse_transform(arr.astype(int))
            return np.asarray(decoded, dtype=object)
        except Exception:
            return values

    def _decoded_classes(self):
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        if self.label_encoder is None:
            return classes
        try:
            return self.label_encoder.inverse_transform(np.asarray(classes).astype(int))
        except Exception:
            return classes


class PrefitVotingEnsemble:
    """Reusable ensemble over already-fitted inference artifacts."""

    def __init__(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
        task_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        class_labels: Optional[List[Any]] = None,
        target_name: str = "",
    ):
        self.models = list(models or [])
        self.weights = [float(weight) for weight in (weights or [1.0] * len(self.models))]
        self.task_type = task_type
        self.target_name = target_name
        self.feature_names_in_ = list(feature_names or [])
        self.classes_ = np.asarray(class_labels or [], dtype=object) if task_type == "classification" else None

    def predict(self, X):
        if self.task_type == "classification":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        stacked = self._stack_regression_predictions(X)
        weighted = np.average(stacked, axis=1, weights=self._resolved_weights())
        return weighted

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise AttributeError("Regression ensembles do not support predict_proba")
        scores = self._classification_scores(X)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return scores / row_sums

    def _resolved_weights(self) -> np.ndarray:
        if len(self.weights) != len(self.models):
            return np.ones(len(self.models), dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        weights[~np.isfinite(weights)] = 1.0
        if float(weights.sum()) <= 0:
            return np.ones(len(self.models), dtype=float)
        return weights

    def _stack_regression_predictions(self, X) -> np.ndarray:
        if not self.models:
            return np.empty((0, 0))
        columns = []
        for model in self.models:
            preds = np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
            columns.append(preds)
        return np.hstack(columns)

    def _classification_scores(self, X) -> np.ndarray:
        classes = list(self.classes_) if self.classes_ is not None else []
        if not classes:
            raise ValueError("Classification ensemble has no class labels.")
        weights = self._resolved_weights()
        scores = None

        for index, model in enumerate(self.models):
            weight = float(weights[index]) if index < len(weights) else 1.0
            current = None

            if hasattr(model, "predict_proba"):
                try:
                    raw = np.asarray(model.predict_proba(X), dtype=float)
                    current = np.zeros((raw.shape[0], len(classes)), dtype=float)
                    model_classes_attr = getattr(model, "classes_", None)
                    model_classes = list(model_classes_attr) if model_classes_attr is not None else []
                    if model_classes and raw.ndim == 2 and raw.shape[1] == len(model_classes):
                        lookup = {label: pos for pos, label in enumerate(model_classes)}
                        for class_index, label in enumerate(classes):
                            source_index = lookup.get(label)
                            if source_index is not None:
                                current[:, class_index] = raw[:, source_index]
                except Exception:
                    current = None

            if current is None:
                preds = np.asarray(model.predict(X), dtype=object).reshape(-1)
                current = np.zeros((preds.shape[0], len(classes)), dtype=float)
                lookup = {label: pos for pos, label in enumerate(classes)}
                for row_index, label in enumerate(preds):
                    class_index = lookup.get(label)
                    if class_index is not None:
                        current[row_index, class_index] = 1.0

            current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            scores = current * weight if scores is None else scores + (current * weight)

        if scores is None:
            raise ValueError("Classification ensemble could not score the input.")
        return scores


class PrefitStackingEnsemble:
    """Stacking ensemble that trains only the meta-model and reuses fitted base artifacts."""

    def __init__(
        self,
        models: List[Any],
        meta_model: Any,
        task_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        class_labels: Optional[List[Any]] = None,
        target_name: str = "",
        meta_feature_names: Optional[List[str]] = None,
    ):
        self.models = list(models or [])
        self.meta_model = meta_model
        self.task_type = task_type
        self.target_name = target_name
        self.feature_names_in_ = list(feature_names or [])
        self.meta_feature_names_ = list(meta_feature_names or [])
        self.classes_ = (
            np.asarray(getattr(meta_model, "classes_", class_labels or []), dtype=object)
            if task_type == "classification"
            else None
        )

    def predict(self, X):
        meta_features = self._build_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise AttributeError("Regression ensembles do not support predict_proba")
        meta_features = self._build_meta_features(X)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        preds = np.asarray(self.meta_model.predict(meta_features), dtype=object).reshape(-1)
        classes = list(self.classes_) if self.classes_ is not None else []
        if not classes:
            raise ValueError("Classification ensemble has no class labels.")
        lookup = {label: pos for pos, label in enumerate(classes)}
        out = np.zeros((preds.shape[0], len(classes)), dtype=float)
        for row_index, label in enumerate(preds):
            class_index = lookup.get(label)
            if class_index is not None:
                out[row_index, class_index] = 1.0
        return out

    def _build_meta_features(self, X) -> pd.DataFrame:
        blocks: List[np.ndarray] = []
        names: List[str] = []

        for model_index, model in enumerate(self.models):
            prefix = f"model_{model_index + 1}"
            if self.task_type == "classification":
                block, block_names = self._classification_block(model, X, prefix)
            else:
                preds = np.asarray(model.predict(X), dtype=float).reshape(-1, 1)
                block = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                block_names = [f"{prefix}_prediction"]
            blocks.append(block)
            names.extend(block_names)

        if not blocks:
            return pd.DataFrame()

        matrix = np.hstack(blocks)
        if self.meta_feature_names_ and len(self.meta_feature_names_) == matrix.shape[1]:
            names = self.meta_feature_names_
        return pd.DataFrame(matrix, columns=names)

    def _classification_block(self, model, X, prefix: str):
        classes = list(self.classes_) if self.classes_ is not None else []
        if not classes:
            raise ValueError("Classification ensemble has no class labels.")

        if hasattr(model, "predict_proba"):
            try:
                raw = np.asarray(model.predict_proba(X), dtype=float)
                aligned = np.zeros((raw.shape[0], len(classes)), dtype=float)
                model_classes_attr = getattr(model, "classes_", None)
                model_classes = list(model_classes_attr) if model_classes_attr is not None else []
                lookup = {label: pos for pos, label in enumerate(model_classes)}
                for class_index, label in enumerate(classes):
                    source_index = lookup.get(label)
                    if source_index is not None:
                        aligned[:, class_index] = raw[:, source_index]
                aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
                return aligned, [f"{prefix}_proba_{label}" for label in classes]
            except Exception:
                pass

        preds = np.asarray(model.predict(X), dtype=object).reshape(-1)
        out = np.zeros((preds.shape[0], len(classes)), dtype=float)
        lookup = {label: pos for pos, label in enumerate(classes)}
        for row_index, label in enumerate(preds):
            class_index = lookup.get(label)
            if class_index is not None:
                out[row_index, class_index] = 1.0
        return out, [f"{prefix}_vote_{label}" for label in classes]
