"""
Legacy frontend validation helpers.

These validators are still useful for Python/Streamlit-side payload handling,
even though the main product UI is now the React studio served by FastAPI.
"""

import logging
from typing import Any, Dict, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Feature data types."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class FeatureValidator:
    """Schema-based feature validation and coercion."""

    # Type detection thresholds
    NUMERIC_THRESHOLD = 0.95  # % of values must parse as numeric
    CATEGORICAL_MAX_UNIQUE = 50  # Max unique values for categorical

    @staticmethod
    def infer_type(values: List[Any]) -> FeatureType:
        """Infer feature type from sample values."""
        if not values:
            return FeatureType.UNKNOWN

        # Remove None/NaN
        clean_vals = [v for v in values if v is not None and str(v).strip()]
        if not clean_vals:
            return FeatureType.UNKNOWN

        # Check boolean
        bool_vals = {str(v).lower() for v in clean_vals}
        if bool_vals.issubset({"true", "false", "1", "0", "yes", "no"}):
            return FeatureType.BOOLEAN

        # Check numeric
        numeric_count = 0
        for v in clean_vals:
            try:
                pd.to_numeric(v)
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        if numeric_count / len(clean_vals) >= FeatureValidator.NUMERIC_THRESHOLD:
            return FeatureType.NUMERIC

        # Check datetime
        datetime_count = 0
        for v in clean_vals:
            try:
                pd.to_datetime(v)
                datetime_count += 1
            except (ValueError, TypeError):
                pass

        if datetime_count / len(clean_vals) > 0.7:
            return FeatureType.DATETIME

        # Default to categorical
        unique_count = len(set(str(v) for v in clean_vals))
        if unique_count <= FeatureValidator.CATEGORICAL_MAX_UNIQUE:
            return FeatureType.CATEGORICAL

        return FeatureType.TEXT

    @staticmethod
    def coerce_numeric(value: Any) -> Union[float, int, None]:
        """Coerce value to numeric (handles 1e-5, " 3 ", NaN, etc.)."""
        if value is None:
            return None

        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isnan(value) if isinstance(value, (float, np.floating)) else False:
                return None
            return float(value)

        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"nan", "none", "null", "na", "n/a"}:
                return None

        try:
            numeric_val = pd.to_numeric(value)
            if np.isnan(numeric_val):
                return None
            return float(numeric_val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def coerce_categorical(value: Any) -> Optional[str]:
        """Coerce value to categorical (string)."""
        if value is None:
            return None

        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None

        return s

    @staticmethod
    def coerce_boolean(value: Any) -> Optional[bool]:
        """Coerce value to boolean."""
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        s = str(value).lower().strip()
        if s in {"true", "1", "yes", "on"}:
            return True
        elif s in {"false", "0", "no", "off"}:
            return False

        return None

    @staticmethod
    def validate_feature_payload(
        features: Dict[str, Any],
        schema: Optional[Dict[str, FeatureType]] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate and coerce feature payload.

        Returns: (processed_features, warnings)
        """
        processed = {}
        warnings = []

        for key, value in (features or {}).items():
            if value is None:
                continue

            # Infer expected type if not provided
            if schema and key in schema:
                expected_type = schema[key]
            else:
                expected_type = FeatureValidator.infer_type([value])

            # Coerce based on type
            try:
                if expected_type == FeatureType.NUMERIC:
                    coerced = FeatureValidator.coerce_numeric(value)
                    if coerced is None and value is not None:
                        warnings.append(
                            f"Could not convert '{key}' to numeric: {value}"
                        )
                    processed[key] = coerced

                elif expected_type == FeatureType.BOOLEAN:
                    coerced = FeatureValidator.coerce_boolean(value)
                    if coerced is None and value is not None:
                        warnings.append(
                            f"Could not convert '{key}' to boolean: {value}"
                        )
                    processed[key] = coerced

                elif expected_type == FeatureType.CATEGORICAL:
                    coerced = FeatureValidator.coerce_categorical(value)
                    if coerced is None and value is not None:
                        warnings.append(f"'{key}' is empty or null")
                    if coerced:  # Only add if not None
                        processed[key] = coerced

                elif expected_type == FeatureType.DATETIME:
                    try:
                        processed[key] = str(pd.to_datetime(value))
                    except Exception:
                        warnings.append(f"Could not parse '{key}' as datetime: {value}")

                else:  # TEXT or UNKNOWN
                    processed[key] = FeatureValidator.coerce_categorical(value)

            except Exception as e:
                logger.warning(f"Error processing feature '{key}': {e}")
                warnings.append(f"Error processing '{key}': {str(e)[:100]}")

        return processed, warnings


class ContractValidator:
    """Validate feature contract alignment."""

    @staticmethod
    def check_alignment(
        expected_features: List[str],
        actual_features: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Check if actual features align with expected schema.

        Returns: {
            "aligned": bool,
            "missing": list,
            "extra": list,
            "warnings": list,
            "valid": bool
        }
        """
        expected = set(expected_features or [])
        actual = set(actual_features.keys() if actual_features else [])

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        warnings = []
        if missing:
            warnings.append(f"Missing features: {', '.join(missing[:5])}")
        if extra:
            warnings.append(f"Extra features: {', '.join(extra[:5])}")

        aligned = len(missing) == 0 and (not strict or len(extra) == 0)
        valid = len(missing) == 0  # Valid if no missing, extra is acceptable

        return {
            "aligned": aligned,
            "valid": valid,
            "missing": missing,
            "extra": extra,
            "warnings": warnings,
        }

    @staticmethod
    def validate_types(
        features: Dict[str, Any],
        schema: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Validate feature types against schema.

        Returns: {
            "valid": bool,
            "mismatches": list,
            "warnings": list
        }
        """
        mismatches = []
        warnings = []

        for feature_name, expected_type in (schema or {}).items():
            if feature_name not in features:
                continue

            value = features[feature_name]
            expected = FeatureType(expected_type)

            # Infer actual type
            actual = FeatureValidator.infer_type([value])

            # Check compatibility (numeric can coerce to numeric, etc.)
            if expected == FeatureType.NUMERIC:
                if FeatureValidator.coerce_numeric(value) is None and value is not None:
                    mismatches.append(
                        {
                            "feature": feature_name,
                            "expected": expected.value,
                            "actual": actual.value,
                            "value": str(value)[:50],
                        }
                    )
                    warnings.append(
                        f"Type mismatch for '{feature_name}': expected numeric"
                    )

        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches,
            "warnings": warnings,
        }


class InputSanitizer:
    """Sanitize user inputs."""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            value = str(value)

        # Remove leading/trailing whitespace
        value = value.strip()

        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"String truncated to {max_length} chars")

        return value

    @staticmethod
    def sanitize_numeric(
        value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> Optional[float]:
        """Sanitize numeric input with bounds."""
        coerced = FeatureValidator.coerce_numeric(value)
        if coerced is None:
            return None

        # Check bounds
        if min_val is not None and coerced < min_val:
            logger.warning(f"Value {coerced} below min {min_val}, clamping")
            coerced = min_val

        if max_val is not None and coerced > max_val:
            logger.warning(f"Value {coerced} above max {max_val}, clamping")
            coerced = max_val

        return coerced

    @staticmethod
    def sanitize_job_id(job_id: str) -> str:
        """Sanitize job ID (prevent injection)."""
        if not job_id or not isinstance(job_id, str):
            raise ValidationError("Invalid job ID")

        # Allow alphanumeric, dash, underscore
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
            raise ValidationError(f"Invalid job ID format: {job_id}")

        if len(job_id) > 100:
            raise ValidationError("Job ID too long")

        return job_id
