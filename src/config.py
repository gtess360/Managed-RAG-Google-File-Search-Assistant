"""Configuration and path management for the Personal AI Knowledge Assistant.

This module provides validated absolute paths and loads environment variables
from a local .env file. Paths are computed relative to the project root so that
the application behaves identically regardless of the current working directory.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Derive the project root folder from this file's location:
# src/config.py -> parent is src/ -> parent of src/ is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the root .env file if it exists.
# override=False preserves any explicitly passed system environment variables.
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable collection of standard absolute paths used throughout the project."""
    root: Path
    data_dir: Path
    storage_dir: Path
    evaluation_dir: Path
    docs_dir: Path


_data_env = os.getenv("DOCUMENTS_DIRECTORY", "").strip()

# Instantiate standard project paths (supporting persistent volume mounts in containers).
# storage_dir only holds the cached File Search store resource name.
PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    data_dir=Path(_data_env) if _data_env else PROJECT_ROOT / "data" / "documents",
    storage_dir=PROJECT_ROOT / "storage",
    evaluation_dir=PROJECT_ROOT / "evaluation",
    docs_dir=PROJECT_ROOT / "docs",
)

# Gemini File Search configuration.
# The model ID must be explicitly configured via GEMINI_MODEL to whichever Gemini model
# is available to your API key and supports File Search. Do not assume any specific model is available.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "").strip()


def has_gemini_model() -> bool:
    """Return True if a Gemini model identifier is configured (non-empty)."""
    return bool((os.getenv("GEMINI_MODEL", "") or GEMINI_MODEL).strip())


def get_gemini_model() -> str:
    """Retrieve the configured Gemini model identifier or raise an actionable error."""
    model = (os.getenv("GEMINI_MODEL", "") or GEMINI_MODEL).strip()
    if not model:
        raise ValueError(
            "Missing required environment variable: GEMINI_MODEL.\n"
            "Action required: Set GEMINI_MODEL in your local '.env' file to a model "
            "enabled for your Google AI / Gemini API account that supports File Search.\n"
            "Check available models in your Google AI Studio account."
        )
    return model


# Display name of the managed File Search store, and where its resource name is cached.
FILE_SEARCH_STORE_DISPLAY_NAME: str = (
    os.getenv("FILE_SEARCH_STORE_DISPLAY_NAME", "personal-ai-knowledge-assistant").strip()
    or "personal-ai-knowledge-assistant"
)
# Optional explicit store resource name (e.g. fileSearchStores/abc123); overrides the cache file.
FILE_SEARCH_STORE_NAME: str = os.getenv("FILE_SEARCH_STORE_NAME", "").strip()
FILE_SEARCH_STATE_FILE: Path = PATHS.storage_dir / "file_search_store.json"


def _gemini_key_from_env() -> str:
    return (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()


def has_gemini_api_key() -> bool:
    """Return True if a Gemini API key is configured (non-empty and not the placeholder)."""
    key = _gemini_key_from_env()
    return bool(key) and key != "your_gemini_api_key_here"


def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key or raise an actionable error."""
    key = _gemini_key_from_env()
    if not key or key == "your_gemini_api_key_here":
        raise ValueError(
            "Missing or placeholder GEMINI_API_KEY.\n"
            "Action required: set GEMINI_API_KEY=<your_actual_key> in the local '.env' file.\n"
            "Note: Never commit .env to version control."
        )
    return key


# Default retrieval candidate volume (shared across backend, API, and evaluator defaults)
# Note: Frontend defaults are separately and statically configured; not dynamically shared.
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5").strip() or "5")

# Gemini request timeout configuration (in seconds)
# Default is 120 seconds to prevent HTTP 504 Gateway Timeout on complex retrieval queries.
DEFAULT_GEMINI_TIMEOUT_SECONDS: float = 120.0


def get_gemini_timeout_seconds() -> float:
    """Retrieve the configured Gemini request timeout in seconds (default: 120.0s)."""
    raw = os.getenv("GEMINI_TIMEOUT_SECONDS")
    if raw is not None and raw.strip():
        try:
            val = float(raw.strip())
            return val if val > 0 else DEFAULT_GEMINI_TIMEOUT_SECONDS
        except ValueError:
            return DEFAULT_GEMINI_TIMEOUT_SECONDS
    return GEMINI_TIMEOUT_SECONDS


GEMINI_TIMEOUT_SECONDS: float = (
    float(os.getenv("GEMINI_TIMEOUT_SECONDS", "120").strip() or "120")
    if os.getenv("GEMINI_TIMEOUT_SECONDS", "").strip()
    else DEFAULT_GEMINI_TIMEOUT_SECONDS
)

# Allowed explicit application environments
VALID_APP_ENVS = frozenset({"development", "production"})


def validate_app_env(env_val: Optional[str] = None) -> str:
    """Validate that APP_ENV is explicitly set to 'development' or 'production'.

    Raises:
        ValueError: If APP_ENV is missing, empty, or set to an unrecognized value.
    """
    raw = env_val if env_val is not None else os.getenv("APP_ENV")
    if raw is None or not raw.strip():
        raise ValueError(
            "Missing required environment variable: APP_ENV.\n"
            "Action required: Explicitly set APP_ENV to 'development' or 'production' in your environment or '.env' file.\n"
            "For local development, add 'APP_ENV=development' to your root .env file."
        )
    cleaned = raw.strip().lower()
    if cleaned not in VALID_APP_ENVS:
        raise ValueError(
            f"Invalid APP_ENV value: '{raw}'.\n"
            f"Allowed values are strictly: {', '.join(sorted(VALID_APP_ENVS))}.\n"
            "Action required: Set APP_ENV=development for local pairing or APP_ENV=production for deployment."
        )
    return cleaned


# Application environment setting
APP_ENV: str = os.getenv("APP_ENV", "").strip().lower()

# Optional Reviewer / Demo Contact Information
CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "").strip()
CONTACT_URL: str = os.getenv("CONTACT_URL", "").strip()



def ensure_directories() -> None:
    """Create essential project directories if they do not already exist."""
    PATHS.data_dir.mkdir(parents=True, exist_ok=True)
    PATHS.evaluation_dir.mkdir(parents=True, exist_ok=True)


def get_safe_config_summary() -> dict:
    """Return a dictionary of configuration settings safe for logging and display.
    
    Never exposes, prints, or logs the actual API key string.
    """
    return {
        "project_root": str(PATHS.root),
        "data_dir": str(PATHS.data_dir),
        "storage_dir": str(PATHS.storage_dir),
        "evaluation_dir": str(PATHS.evaluation_dir),
        "gemini_model": GEMINI_MODEL or "Not configured",
        "gemini_api_key_configured": has_gemini_api_key(),
        "file_search_store_display_name": FILE_SEARCH_STORE_DISPLAY_NAME,
        "app_env": APP_ENV,
        "gemini_timeout_seconds": get_gemini_timeout_seconds(),
        "contact_configured": bool(os.getenv("CONTACT_EMAIL", "").strip() or os.getenv("CONTACT_URL", "").strip()),
    }
