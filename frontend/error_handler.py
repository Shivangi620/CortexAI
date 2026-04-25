"""
Legacy frontend error handling helpers.

These utilities support older Python/Streamlit-facing workflows in `frontend/`.
The primary user interface is now the React studio, but these helpers still
provide structured error categorization for compatibility codepaths.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Error classification for frontend -> backend communication."""

    NETWORK = "NETWORK"  # Connection, timeout
    VALIDATION = "VALIDATION"  # Bad input, schema mismatch
    BACKEND = "BACKEND"  # Server error
    AUTH = "AUTH"  # Authentication/authorization
    NOT_FOUND = "NOT_FOUND"  # 404
    RATE_LIMIT = "RATE_LIMIT"  # 429
    UNKNOWN = "UNKNOWN"  # Uncategorized


@dataclass
class APIError:
    """Structured error response."""

    type: ErrorType
    message: str
    retryable: bool
    status_code: Optional[int] = None
    original_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error": {
                "type": self.type.value,
                "message": self.message,
                "retryable": self.retryable,
                "status_code": self.status_code,
            }
        }

    def user_message(self) -> str:
        """Get user-friendly error message."""
        messages = {
            ErrorType.NETWORK: "Connection issue. Check your network and try again.",
            ErrorType.VALIDATION: "Invalid input. Please check your data format.",
            ErrorType.BACKEND: "Server error. Try again in a moment.",
            ErrorType.AUTH: "Authentication failed. Please log in again.",
            ErrorType.NOT_FOUND: "Resource not found. It may have been deleted.",
            ErrorType.RATE_LIMIT: "Too many requests. Please wait before trying again.",
            ErrorType.UNKNOWN: "An unexpected error occurred.",
        }
        return messages.get(self.type, self.message)


class ErrorHandler:
    """Centralized error handling with logging."""

    @classmethod
    def from_requests_exception(cls, e: Exception, context: str = "") -> APIError:
        """Convert requests library exception to APIError."""
        error_msg = str(e)

        if "timeout" in error_msg.lower() or "connect" in error_msg.lower():
            error_type = ErrorType.NETWORK
            retryable = True
        else:
            error_type = ErrorType.NETWORK
            retryable = True

        logger.error(f"Request error in {context}: {error_msg}", exc_info=True)
        return APIError(
            type=error_type,
            message=error_msg[:200],
            retryable=retryable,
            original_error=error_msg,
        )

    @classmethod
    def from_http_status(
        cls,
        status_code: int,
        response_text: str = "",
        context: str = "",
    ) -> APIError:
        """Convert HTTP status code to APIError."""
        status_map = {
            400: (ErrorType.VALIDATION, False),
            401: (ErrorType.AUTH, False),
            403: (ErrorType.AUTH, False),
            404: (ErrorType.NOT_FOUND, False),
            429: (ErrorType.RATE_LIMIT, True),
            500: (ErrorType.BACKEND, True),
            502: (ErrorType.BACKEND, True),
            503: (ErrorType.BACKEND, True),
        }

        error_type, retryable = status_map.get(
            status_code,
            (ErrorType.BACKEND, True if status_code >= 500 else False),
        )

        message = f"HTTP {status_code}"
        if response_text:
            try:
                import json

                data = json.loads(response_text)
                message = data.get("detail", data.get("error", message))[:200]
            except Exception:
                message = response_text[:200]

        logger.warning(f"HTTP {status_code} in {context}: {message}")

        return APIError(
            type=error_type,
            message=message,
            retryable=retryable,
            status_code=status_code,
        )

    @classmethod
    def validation_error(cls, field: str, reason: str) -> APIError:
        """Create validation error."""
        message = f"Validation failed for '{field}': {reason}"
        logger.warning(message)
        return APIError(
            type=ErrorType.VALIDATION,
            message=message,
            retryable=False,
        )

    @classmethod
    def from_dict(cls, error_dict: Dict[str, Any], context: str = "") -> APIError:
        """Create APIError from error dict."""
        if "error" in error_dict:
            if isinstance(error_dict["error"], dict):
                err = error_dict["error"]
                error_type = ErrorType(err.get("type", "UNKNOWN"))
                message = err.get("message", "Unknown error")
                retryable = err.get("retryable", True)
                status_code = err.get("status_code")
            else:
                error_type = ErrorType.UNKNOWN
                message = str(error_dict["error"])
                retryable = True
                status_code = None
        else:
            error_type = ErrorType.UNKNOWN
            message = "Unknown error"
            retryable = True
            status_code = None

        logger.warning(f"Error in {context}: [{error_type.value}] {message}")

        return APIError(
            type=error_type,
            message=message,
            retryable=retryable,
            status_code=status_code,
        )

    @classmethod
    def should_retry(cls, error: APIError, attempt: int = 1) -> Tuple[bool, int]:
        """
        Determine if error should be retried and wait time.

        Returns: (should_retry, wait_seconds)
        """
        if not error.retryable:
            return False, 0

        if attempt > 3:
            return False, 0

        # Exponential backoff: 1s, 2s, 4s
        wait_time = min(2 ** (attempt - 1), 4)
        return True, wait_time

    @classmethod
    def log_api_call(
        cls,
        method: str,
        endpoint: str,
        status: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log API call for observability."""
        if status is None:
            logger.info(f"API {method} {endpoint}")
        elif 200 <= status < 300:
            logger.info(
                f"API {method} {endpoint} → {status}"
                + (f" ({duration_ms:.0f}ms)" if duration_ms else "")
            )
        else:
            logger.warning(
                f"API {method} {endpoint} → {status}"
                + (f" ({duration_ms:.0f}ms)" if duration_ms else "")
            )
