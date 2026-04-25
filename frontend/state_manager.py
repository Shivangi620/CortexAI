"""
Legacy Streamlit state management layer.

This module backs older Streamlit-oriented results and playground flows.
The current production UI is the React studio, but these state containers
remain useful for compatibility and migration-safe support.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class PlaygroundState:
    """Playground tab state."""

    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    live_whatif_enabled: bool = True
    last_prediction: Optional[Dict[str, Any]] = None
    last_explanation: Optional[Dict[str, Any]] = None
    contract_check_result: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisState:
    """Analysis tab state."""

    view_mode: str = "Advanced"  # "Beginner" or "Advanced"
    selected_drift_feature: Optional[str] = None
    calibration_cache: Optional[Dict[str, Any]] = None
    thresholds_cache: Optional[Dict[str, Any]] = None


@dataclass
class AugmentationState:
    """Augmentation tab state."""

    synthetic_result: Optional[Dict[str, Any]] = None
    judge_result: Optional[Dict[str, Any]] = None


@dataclass
class AppState:
    """
    Single source of truth for legacy Streamlit results-console state.

    Enforces schema, prevents silent overwrites, and manages lifecycle.
    All state access goes through this dataclass.
    """

    # Core identifiers
    job_id: str = ""
    dataset_id: Optional[str] = None

    # API responses (cached)
    job_data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    insights: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)

    # State for each tab
    playground: PlaygroundState = field(default_factory=PlaygroundState)
    analysis: AnalysisState = field(default_factory=AnalysisState)
    augmentation: AugmentationState = field(default_factory=AugmentationState)

    # Polling state
    poll_count: int = 0
    last_poll_timestamp: Optional[float] = None

    # UI state
    comparison_job_id: Optional[str] = None
    comparison_enabled: bool = False
    comparison_data: Optional[Dict[str, Any]] = None

    # Model chat history
    chat_messages: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)


class StateManager:
    """Centralized state manager with schema enforcement."""

    SESSION_KEY = "_app_state"

    @classmethod
    def initialize(cls) -> AppState:
        """Initialize or retrieve app state from session."""
        if cls.SESSION_KEY not in st.session_state:
            state = AppState()
            st.session_state[cls.SESSION_KEY] = state
            logger.info("Initialized new AppState")
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def get(cls) -> AppState:
        """Get current app state."""
        if cls.SESSION_KEY not in st.session_state:
            return cls.initialize()
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def update_job(cls, job_id: str, job_data: Dict[str, Any]) -> None:
        """Update job context."""
        state = cls.get()
        state.job_id = job_id
        state.job_data = job_data
        state.results = job_data.get("results", {}) or {}
        state.insights = job_data.get("insights", {}) or {}
        logger.info(f"Updated job context: {job_id[:8]}")

    @classmethod
    def update_profile(
        cls, profile: Dict[str, Any], dataset_id: Optional[str] = None
    ) -> None:
        """Update dataset profile."""
        state = cls.get()
        state.profile = profile
        if dataset_id:
            state.dataset_id = dataset_id
        logger.debug(f"Updated profile with {len(profile)} keys")

    @classmethod
    def set_view_mode(cls, mode: str) -> None:
        """Set beginner/advanced view mode."""
        state = cls.get()
        if mode not in ("Beginner", "Advanced"):
            logger.warning(f"Invalid view mode: {mode}, using Advanced")
            mode = "Advanced"
        state.analysis.view_mode = mode
        logger.info(f"View mode changed to: {mode}")

    @classmethod
    def save_scenario(cls, name: str, features: Dict[str, Any]) -> None:
        """Save a playground scenario."""
        import time

        state = cls.get()
        state.playground.scenarios[name] = {
            "features": features.copy(),
            "timestamp": time.time(),
        }
        logger.info(f"Saved scenario: {name}")

    @classmethod
    def load_scenario(cls, name: str) -> Optional[Dict[str, Any]]:
        """Load a saved scenario."""
        state = cls.get()
        scenario = state.playground.scenarios.get(name)
        if scenario:
            logger.info(f"Loaded scenario: {name}")
            return scenario["features"]
        return None

    @classmethod
    def clear_scenarios(cls) -> None:
        """Clear all saved scenarios."""
        state = cls.get()
        count = len(state.playground.scenarios)
        state.playground.scenarios.clear()
        logger.info(f"Cleared {count} scenarios")

    @classmethod
    def set_polling_state(cls, count: int, timestamp: Optional[float] = None) -> None:
        """Update polling state."""
        import time

        state = cls.get()
        state.poll_count = count
        state.last_poll_timestamp = timestamp or time.time()

    @classmethod
    def set_comparison_context(cls, job_id: str, data: Dict[str, Any]) -> None:
        """Set comparison job context."""
        state = cls.get()
        state.comparison_job_id = job_id
        state.comparison_data = data
        logger.info(f"Comparison context set for: {job_id[:8]}")

    @classmethod
    def cache_prediction(cls, prediction: Dict[str, Any]) -> None:
        """Cache last prediction."""
        state = cls.get()
        state.playground.last_prediction = prediction

    @classmethod
    def cache_explanation(cls, explanation: Dict[str, Any]) -> None:
        """Cache last explanation."""
        state = cls.get()
        state.playground.last_explanation = explanation

    @classmethod
    def cache_contract_check(cls, result: Dict[str, Any]) -> None:
        """Cache contract check result."""
        state = cls.get()
        state.playground.contract_check_result = result

    @classmethod
    def add_chat_message(cls, job_id: str, role: str, content: str) -> None:
        """Add message to chat history."""
        state = cls.get()
        if job_id not in state.chat_messages:
            state.chat_messages[job_id] = []
        state.chat_messages[job_id].append({"role": role, "content": content})
        logger.debug(f"Chat message added for job {job_id[:8]}")

    @classmethod
    def get_chat_history(cls, job_id: str) -> List[Dict[str, str]]:
        """Get chat history for job."""
        state = cls.get()
        return state.chat_messages.get(job_id, [])

    @classmethod
    def reset(cls) -> None:
        """Reset to clean state."""
        st.session_state[cls.SESSION_KEY] = AppState()
        logger.info("AppState reset to clean state")

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export state as dictionary (for debugging)."""
        state = cls.get()
        return asdict(state)
