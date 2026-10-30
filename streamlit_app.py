"""Streamlit web interface for the Personal AI Knowledge Assistant.

Directly calls src.pipeline.query_pipeline without requiring FastAPI or Next.js.
Supports Streamlit Community Cloud deployment with:
- Pre-import Streamlit secrets propagation to os.environ (preserving local .env)
- Explicit STREAMLIT_DEMO_BOOTSTRAP mode for automatic missing-index initialization
- Thread-locked, double-checked, once-per-process bootstrap guard
- Session-based 5-free-questions demo allowance before requiring REVIEWER_ACCESS_TOKEN
- Reviewer access token unlock with constant-time verification bypassing demo quota
- Contact link via CONTACT_EMAIL or CONTACT_URL with explicit fallback setup notice
- Configurable candidate retrieval (top_k between 1 and 10, default 5)
- Character-limited question input (<= 500 characters)
- Explicit form submission with session-state caching across UI reruns
- Clear rendering of status (answered, partial, conflict, abstain, error)
- Grounded answers, missing information, evidence-grounded citations, and expandable evidence
- Latency timing breakdowns (retrieval, generation, total)
- Modern portfolio visual design with light background, dark navy typography, and teal accents
"""

from __future__ import annotations

import html
import hmac
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Optional
import streamlit as st

logger = logging.getLogger("streamlit_app")


# -----------------------------------------------------------------------------
# 1. Streamlit Cloud Secrets Propagation (Pre-Config Import)
# -----------------------------------------------------------------------------

def _sync_streamlit_secrets_to_env() -> None:
    """Propagate Streamlit Cloud secrets to os.environ before config imports.

    Streamlit Cloud secrets override local defaults while preserving
    any environment variables not defined in st.secrets.
    Never prints or logs secret values.
    """
    try:
        if hasattr(st, "secrets"):
            for key, val in st.secrets.items():
                if isinstance(val, (str, int, float, bool)):
                    os.environ[key] = str(val)
    except Exception:
        pass


_sync_streamlit_secrets_to_env()

# Now import configuration and pipeline modules (which read environment variables)
from src.config import (
    PATHS,
    GEMINI_MODEL,
    DEFAULT_TOP_K,
    get_safe_config_summary,
    has_gemini_api_key,
)
import src.pipeline as pipeline
import src.file_search as file_search


# -----------------------------------------------------------------------------
# 2. Process-Level Demo Bootstrap Mechanism
# -----------------------------------------------------------------------------

_bootstrap_lock = threading.Lock()
_bootstrap_attempted: bool = False
_bootstrap_error: Optional[str] = None


def is_bootstrap_enabled() -> bool:
    """Check if demo bootstrap mode is explicitly enabled."""
    val = os.getenv("STREAMLIT_DEMO_BOOTSTRAP", "").strip().lower()
    return val in ("true", "1", "yes", "enabled")


def reset_bootstrap_state_for_tests() -> None:
    """Reset process-level bootstrap tracking flags (used only in tests)."""
    global _bootstrap_attempted, _bootstrap_error
    with _bootstrap_lock:
        _bootstrap_attempted = False
        _bootstrap_error = None


def maybe_bootstrap_index(
    storage_dir: Optional[Path] = None,  # deprecated, ignored
    data_dir: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    """Upload demo documents to Gemini File Search if STREAMLIT_DEMO_BOOTSTRAP is enabled and the store is empty.

    Concurrency & Safety Guarantees:
    - Thread-safe: Uses process-level lock and double-checked inspection.
    - Idempotent: Executes at most once per process.
    - Only bootstraps if the store is missing or has no active documents.
    - Never replaces an existing populated store.
    - Blocks subsequent queries if bootstrap fails.

    Returns:
        tuple[bool, Optional[str]]: (success, error_message_if_any)
    """
    global _bootstrap_attempted, _bootstrap_error

    if not is_bootstrap_enabled():
        return True, None

    if _bootstrap_error is not None:
        return False, _bootstrap_error

    target_data = data_dir or PATHS.data_dir

    # 1. First store check (outside lock)
    ready, _msg, _count = file_search.get_store_status()
    if ready:
        return True, None

    # 2. Acquire lock for process-level single initialization
    with _bootstrap_lock:
        if _bootstrap_attempted:
            if _bootstrap_error is not None:
                return False, _bootstrap_error
            return True, None

        # 3. Second store check (inside lock)
        ready, _msg, _count = file_search.get_store_status()
        if ready:
            _bootstrap_attempted = True
            return True, None

        # 4. Upload committed sanitized files
        try:
            from ingest import ingest_corpus
            ingest_corpus(
                allow_partial_ingestion=False,
                data_dir=target_data,
                replace=False,
            )
            _bootstrap_attempted = True
            return True, None
        except Exception as exc:
            _bootstrap_attempted = True
            _bootstrap_error = f"STREAMLIT_DEMO_BOOTSTRAP failed: {exc}"
            logger.error(
                "STREAMLIT_DEMO_BOOTSTRAP failed: %s\nTraceback:\n%s",
                exc,
                traceback.format_exc(),
            )
            return False, _bootstrap_error


# -----------------------------------------------------------------------------
# 3. Read-Only Index Health Check
# -----------------------------------------------------------------------------

def check_index_health(storage_dir: Optional[Path] = None) -> tuple[bool, str]:
    """Check if the Gemini File Search store exists and has active documents.

    Per architecture guidelines, this inspection is strictly read-only.
    It NEVER uploads or alters documents automatically.

    Returns:
        tuple[bool, str]: (is_healthy, message)
    """
    ready, message, _count = file_search.get_store_status()
    return ready, message


# -----------------------------------------------------------------------------
# 4. Demo Quota & Reviewer Token Verification
# -----------------------------------------------------------------------------

FREE_QUESTION_LIMIT: int = 5


def get_contact_destination() -> tuple[Optional[str], Optional[str]]:
    """Return (destination_type, destination_value) or (None, None).

    Reads destination strictly from CONTACT_URL or CONTACT_EMAIL.
    Never invents contact details.
    """
    url = os.getenv("CONTACT_URL", "").strip()
    if url:
        return "url", url
    email = os.getenv("CONTACT_EMAIL", "").strip()
    if email:
        return "email", f"mailto:{email}?subject=Access%20Token%20Request"
    return None, None


def render_request_access_link() -> None:
    """Render the 'Request access' link or an explicit configuration setup notice."""
    dest_type, dest_val = get_contact_destination()
    if dest_type and dest_val:
        st.markdown(f"👉 [**Request access**]({dest_val})")
    else:
        st.info("ℹ️ Contact setup notice: Neither `CONTACT_EMAIL` nor `CONTACT_URL` is configured in the environment.")


def verify_reviewer_token(token_input: str) -> bool:
    """Verify reviewer access token using constant-time comparison.

    Fails closed if REVIEWER_ACCESS_TOKEN is missing or empty.
    """
    expected_token = os.getenv("REVIEWER_ACCESS_TOKEN", "").strip()
    if not expected_token or not token_input:
        return False
    return hmac.compare_digest(token_input.strip(), expected_token)


def get_available_documents() -> list[str]:
    """Return sorted filenames of actual local documents available in the corpus directory.

    Strictly reads real files on disk without hardcoding or fabricating entries.
    """
    if not PATHS.data_dir.exists():
        return []
    docs = [f.name for f in PATHS.data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    return sorted(docs)


def check_production_access() -> bool:
    """Backwards-compatible helper.

    Replaced by session-based 5-free-questions quota flow. Always returns True
    to permit the 5-question allowance to proceed without an immediate blocking gate.
    """
    return True


# -----------------------------------------------------------------------------
# 5. Modern SaaS Dashboard CSS Styling (Phase 1 Foundation)
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Modern SaaS AI Dashboard Foundation */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --navy-950: #090d16;
    --navy-900: #0f172a;
    --navy-800: #1e293b;
    --navy-700: #334155;
    --navy-600: #475569;
    --navy-500: #64748b;
    --navy-400: #94a3b8;
    --teal-700: #0f766e;
    --teal-600: #0d9488;
    --teal-500: #14b8a6;
    --teal-400: #2dd4bf;
    --teal-100: #ccfbf1;
    --teal-50: #f0fdfa;
    --green-600: #16a34a;
    --green-500: #22c55e;
    --green-100: #dcfce7;
    --green-50: #f0fdf4;
    --gray-50: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-300: #cbd5e1;
    --card-bg: #ffffff;
    --shadow-xs: 0 1px 2px 0 rgba(15, 23, 42, 0.04);
    --shadow-sm: 0 1px 3px 0 rgba(15, 23, 42, 0.06), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 4px 12px -2px rgba(15, 23, 42, 0.07), 0 2px 6px -2px rgba(15, 23, 42, 0.04);
}

/* Base Body & Container Adjustments */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--navy-800);
}

.stApp {
    background: #f8fafc !important;
    color: var(--navy-800);
}

/* Streamlit Header & Spacing Adjustments */
header[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 10;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}

/* Sidebar SaaS Styling Foundation */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--gray-200) !important;
    box-shadow: var(--shadow-xs);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.15rem !important;
    padding-right: 1.15rem !important;
}

/* Sidebar Button Styling */
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    box-shadow: 0 2px 5px rgba(13, 148, 136, 0.25) !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
    box-shadow: 0 4px 10px rgba(13, 148, 136, 0.35) !important;
}

/* Sidebar Navigation Items */
.sidebar-nav-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 12px;
    margin-bottom: 16px;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--navy-700);
    text-decoration: none;
    transition: background 0.15s ease;
}

.sidebar-nav-item.active {
    background: #f0fdfa;
    color: var(--teal-700);
    font-weight: 600;
}

.sidebar-nav-item svg {
    color: inherit;
    flex-shrink: 0;
}

/* Sidebar Documents Section */
.sidebar-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--navy-500);
    margin-top: 14px;
    margin-bottom: 8px;
    padding: 0 2px;
}

.sidebar-doc-badge {
    background: var(--gray-200);
    color: var(--navy-700);
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 9999px;
    font-weight: 600;
}

.sidebar-doc-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 16px;
}

.sidebar-doc-item-all {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--navy-800);
    background: var(--gray-100);
    margin-bottom: 2px;
}

.sidebar-doc-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    color: var(--navy-600);
    transition: background 0.15s ease, color 0.15s ease;
}

.sidebar-doc-item:hover {
    background: var(--gray-100);
    color: var(--teal-700);
}

.sidebar-doc-item svg, .sidebar-doc-item-all svg {
    color: var(--navy-400);
    flex-shrink: 0;
}

/* Sidebar Demo Usage Card */
.sidebar-usage-card {
    background: #ffffff;
    border: 1px solid var(--gray-200);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 8px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-xs);
}

.sidebar-usage-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.sidebar-usage-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--navy-800);
    display: flex;
    align-items: center;
    gap: 6px;
}

.info-tooltip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: var(--gray-100);
    color: var(--navy-500);
    font-size: 0.72rem;
    cursor: help;
}

.sidebar-progress-track {
    width: 100%;
    height: 6px;
    background: var(--gray-200);
    border-radius: 9999px;
    overflow: hidden;
    margin-bottom: 8px;
}

.sidebar-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%);
    border-radius: 9999px;
    transition: width 0.3s ease;
}

.sidebar-usage-text {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--navy-800);
    margin-bottom: 2px;
}

.sidebar-usage-caption {
    font-size: 0.72rem;
    color: var(--navy-500);
}

/* Sidebar Expanders (System Status, Advanced Configuration, Reviewer Access) */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 10px !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] details summary {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--navy-800) !important;
    padding: 10px 12px !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 10px 12px 14px 12px !important;
}

/* -------------------------------------------------------------
   Top Navigation Bar Component
   ------------------------------------------------------------- */
.saas-top-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid var(--gray-200);
    border-radius: 14px;
    padding: 12px 20px;
    margin-top: 20px;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
}

.saas-brand-identity {
    display: flex;
    align-items: center;
    gap: 14px;
}

.saas-brand-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    color: #ffffff;
    box-shadow: 0 4px 10px rgba(13, 148, 136, 0.25);
    flex-shrink: 0;
}

.saas-brand-text {
    display: flex;
    flex-direction: column;
}

.saas-brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--navy-900);
    line-height: 1.25;
    letter-spacing: -0.01em;
}

.saas-brand-subtitle {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--navy-500);
    line-height: 1.35;
    margin-top: 2px;
}

.saas-navbar-utilities {
    display: flex;
    align-items: center;
}

/* Live Status Indicator Pill */
.saas-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 9999px;
    transition: all 0.2s ease;
}

.saas-status-badge.online {
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.saas-status-badge.offline {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}

/* Main Heading Integration */
.main-title-wrapper {
    display: none; /* Accessible to tests while visually rendered by the top brand bar */
}

/* Visually hide default Streamlit h1 so only modern hero title shows */
h1:not(.hero-main-title) {
    display: none !important;
}

/* Header Badge & Subtitle for Backward & Test Compatibility */
.header-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--teal-700);
    background: var(--teal-100);
    border: 1px solid rgba(13, 148, 136, 0.25);
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 4px;
}

.header-subtitle {
    color: var(--navy-600);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: -8px;
    margin-bottom: 12px;
    line-height: 1.5;
}

.header-motif-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: linear-gradient(90deg, rgba(13, 148, 136, 0.08) 0%, rgba(2, 132, 199, 0.08) 100%);
    border: 1px solid rgba(13, 148, 136, 0.15);
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 0.85rem;
    color: var(--teal-700);
    font-weight: 500;
}

/* Allowance Badges */
.allowance-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: var(--teal-50);
    color: var(--teal-700);
    border: 1px solid #99f6e4;
    margin-bottom: 8px;
}

.allowance-badge.exhausted {
    background: #fff1f2;
    color: #be123c;
    border-color: #fecdd3;
}

.allowance-badge.unlocked {
    background: #f0fdf4;
    color: #15803d;
    border-color: #bbf7d0;
}

/* Primary Teal Submit Action */
button[kind="primary"] {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 9px 24px !important;
    box-shadow: 0 2px 4px rgba(13, 148, 136, 0.2) !important;
    transition: all 0.2s ease-in-out !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35) !important;
}

/* Secondary Example Buttons */
button[kind="secondary"] {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #334155 !important;
    transition: all 0.15s ease-in-out !important;
}

button[kind="secondary"]:hover {
    border-color: #0d9488 !important;
    color: #0f766e !important;
    background: #f0fdfa !important;
}

/* Focus states */
textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 1px #0d9488 !important;
}

/* -------------------------------------------------------------
   Phase 3: Modern Hero Section & Suggested Cards
   ------------------------------------------------------------- */
.saas-hero-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #f0fdfa 0%, #f8fafc 100%);
    border: 1px solid rgba(13, 148, 136, 0.2);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 22px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}

.hero-content {
    max-width: 680px;
}

.hero-badge-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--teal-700);
    background: var(--teal-100);
    border: 1px solid rgba(13, 148, 136, 0.25);
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 8px;
}

.hero-main-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--navy-900);
    line-height: 1.2;
    margin: 4px 0 8px 0;
    letter-spacing: -0.02em;
}

.hero-main-desc {
    font-size: 0.95rem;
    color: var(--navy-600);
    line-height: 1.5;
    margin: 0;
}

.hero-illustration-box {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-left: 20px;
}

/* Suggested Questions Section */
.suggested-section-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.suggested-section-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--navy-900);
    display: flex;
    align-items: center;
    gap: 6px;
}

.suggested-section-link {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--teal-600);
    text-decoration: none;
}

/* Suggested Question Card Buttons */
[data-testid="column"] button[kind="secondary"] {
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--gray-200) !important;
    background: #ffffff !important;
    color: var(--navy-800) !important;
    padding: 12px 14px !important;
    box-shadow: var(--shadow-xs) !important;
    text-align: left !important;
    min-height: 72px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    line-height: 1.4 !important;
    transition: all 0.2s ease !important;
}

[data-testid="column"] button[kind="secondary"]:hover {
    border-color: var(--teal-500) !important;
    color: var(--teal-700) !important;
    background: #f0fdfa !important;
    box-shadow: var(--shadow-sm) !important;
    transform: translateY(-1px);
}

/* Explore Examples Toggle Button in Header Bar */
.explore-btn-container button {
    min-height: unset !important;
    height: 32px !important;
    padding: 4px 12px !important;
    border-radius: 9999px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--teal-700) !important;
    background: var(--teal-50) !important;
    border: 1px solid var(--teal-200) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
}

.explore-btn-container button:hover {
    background: var(--teal-100) !important;
    border-color: var(--teal-500) !important;
    color: var(--teal-800) !important;
    transform: none !important;
}

/* -------------------------------------------------------------
   Phase 3: Chat Workspace Cards
   ------------------------------------------------------------- */
.chat-workspace-wrap {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 20px;
}

.chat-message-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}

.chat-avatar-user {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--navy-800);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: var(--shadow-xs);
}

.chat-bubble-user {
    background: #f1f5f9;
    border: 1px solid var(--gray-200);
    border-radius: 14px 14px 14px 2px;
    padding: 14px 18px;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chat-bubble-user-text {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--navy-900);
    line-height: 1.45;
}

.chat-bubble-user-meta {
    font-size: 0.75rem;
    color: var(--navy-400);
    margin-left: 12px;
    flex-shrink: 0;
}

.chat-avatar-assistant {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 3px 8px rgba(13, 148, 136, 0.25);
}

.chat-bubble-assistant {
    background: #ffffff;
    border: 1px solid var(--gray-200);
    border-radius: 14px;
    padding: 20px 22px;
    flex-grow: 1;
    box-shadow: var(--shadow-sm);
}

.assistant-answer-text {
    font-size: 0.96rem;
    color: var(--navy-900);
    line-height: 1.65;
    margin-bottom: 12px;
}

/* Horizontal Overflow and Wrapping Protections */
.stMarkdown, .stText, p, span, code, pre, [data-testid="stExpander"] {
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}

code {
    word-break: break-all !important;
    white-space: pre-wrap !important;
}

pre {
    white-space: pre-wrap !important;
    overflow-x: auto !important;
    max-width: 100% !important;
}

/* Ensure containers and cards fit desktop and mobile viewports */
[data-testid="stExpander"], [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"], .block-container, .chat-workspace-wrap {
    max-width: 100% !important;
    box-sizing: border-box !important;
}

.chat-bubble-user, .chat-bubble-assistant {
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}

/* Assistant Header in Container */
.assistant-row {
    margin-bottom: 8px !important;
}

.chat-assistant-header-text {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0 0 4px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--navy-900) !important;
}

.chat-assistant-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--navy-900);
}

/* Native Sources Expander Styling in Main Workspace */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] {
    background: #f8fafc !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
    margin-bottom: 8px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] details summary {
    font-size: 0.86rem !important;
    font-weight: 700 !important;
    color: var(--navy-800) !important;
    padding: 8px 12px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 8px 14px 12px 14px !important;
    background: #ffffff !important;
    border-top: 1px solid var(--gray-200) !important;
}

/* Form Container Card */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid var(--gray-200);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: var(--shadow-sm);
    margin-top: 14px;
}

[data-testid="stForm"] textarea {
    border-radius: 10px !important;
    border: 1px solid var(--gray-300) !important;
    font-size: 0.95rem !important;
    padding: 12px 14px !important;
}

[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    box-shadow: 0 2px 6px rgba(13, 148, 136, 0.25) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
    background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35) !important;
}

/* Responsive Media Queries */
@media (max-width: 768px) {
    .saas-top-navbar {
        margin-top: 16px;
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
        padding: 12px 16px;
        margin-bottom: 1.25rem;
    }

    .saas-brand-title {
        font-size: 1.1rem;
    }

    .saas-hero-card {
        flex-direction: column;
        align-items: flex-start;
        padding: 18px 20px;
    }

    .hero-illustration-box {
        display: none;
    }

    .hero-main-title {
        font-size: 1.45rem;
    }

    .chat-bubble-assistant {
        padding: 16px;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


# -----------------------------------------------------------------------------
# 6. Sidebar Component (Rendered after state updates to ensure zero-latency sync)
# -----------------------------------------------------------------------------

def render_sidebar(is_healthy: bool, health_msg: str) -> None:
    """Render the sidebar after state mutations so history and quota updates appear immediately."""
    is_authenticated = st.session_state.get("authenticated", False)
    questions_used = st.session_state.get("questions_used", 0)
    remaining = max(0, FREE_QUESTION_LIMIT - questions_used)

    with st.sidebar:
        # 1. New Conversation Action Button
        # Clears active answer card and question input, preserves session quota and history
        if st.button("➕ New Conversation", key="btn_new_conversation", type="primary", use_container_width=True, help="Start a new question thread (preserves session quota and history)"):
            st.session_state["last_result"] = None
            st.session_state["question_input"] = ""
            st.rerun()

        # 2. Navigation items with Recent Questions
        st.markdown(
            """
            <div class="sidebar-nav-list">
                <div class="sidebar-nav-item active">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                    <span>Home</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🕒 Recent Questions", expanded=False):
            recent_qs = st.session_state.get("recent_questions", [])
            if recent_qs:
                for q_i, rq in enumerate(recent_qs):
                    if st.button(f"💬 {rq}", key=f"sidebar_recent_q_{q_i}", help=rq, use_container_width=True):
                        st.session_state["question_input"] = rq
                        st.rerun()
            else:
                st.caption("No recent questions submitted in this session.")

        # 3. Documents section (Actual available documents with source description)
        available_docs = get_available_documents()
        doc_count = len(available_docs)

        docs_list_html = "".join([
            f'<div class="sidebar-doc-item">'
            f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>'
            f'<span>{html.escape(doc)}</span>'
            f'</div>'
            for doc in available_docs
        ])

        st.markdown(
            f"""
            <div class="sidebar-section-header">
                <span>DOCUMENTS</span>
                <span class="sidebar-doc-badge">{doc_count}</span>
            </div>
            <div class="sidebar-doc-list">
                <div class="sidebar-doc-item-all">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="8" height="18" x="3" y="3" rx="1"></rect><path d="m15 3 6 6v12a1 1 0 0 1-1 1h-5"></path></svg>
                    <span>All Documents ({doc_count})</span>
                </div>
                <div style="font-size: 0.72rem; color: var(--navy-500); padding: 2px 10px 6px 10px;">
                    Corpus: <code>data/documents/</code>
                </div>
                {docs_list_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Demo Usage card with progress bar
        progress_ratio = min(1.0, max(0.0, (FREE_QUESTION_LIMIT - remaining) / float(FREE_QUESTION_LIMIT)))
        progress_pct = int(progress_ratio * 100)

        if is_authenticated:
            st.markdown(
                """
                <div class="sidebar-usage-card">
                    <div class="sidebar-usage-header">
                        <span class="sidebar-usage-title">
                            Demo Usage
                            <span class="info-tooltip" title="Session-based demo allowance">ⓘ</span>
                        </span>
                    </div>
                    <div class="sidebar-progress-track">
                        <div class="sidebar-progress-fill" style="width: 100%;"></div>
                    </div>
                    <div class="sidebar-usage-text" style="color: #15803d;">🔓 Unlimited Access</div>
                    <div class="sidebar-usage-caption">Reviewer token active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="sidebar-usage-card">
                    <div class="sidebar-usage-header">
                        <span class="sidebar-usage-title">
                            Demo Usage
                            <span class="info-tooltip" title="Session-based demo allowance">ⓘ</span>
                        </span>
                    </div>
                    <div class="sidebar-progress-track">
                        <div class="sidebar-progress-fill" style="width: {progress_pct}%;"></div>
                    </div>
                    <div class="sidebar-usage-text">{remaining} of {FREE_QUESTION_LIMIT} free questions remaining</div>
                    <div class="sidebar-usage-caption">Session-based demo allowance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. Reviewer Access section (collapsible card preserving form inputs and Unlock Token button)
        with st.expander("🔒 Reviewer Access", expanded=not is_authenticated):
            if is_authenticated:
                st.success("Reviewer Token: Active (Unlimited)")
                if st.button("Relock Assistant", key="sidebar_relock_btn", use_container_width=True):
                    st.session_state["authenticated"] = False
                    st.session_state["last_result"] = None
                    st.rerun()
            else:
                with st.form("sidebar_reviewer_token_form", clear_on_submit=True):
                    token_val = st.text_input(
                        "Reviewer Access Token",
                        type="password",
                        placeholder="Enter access token...",
                        help="Unlock unlimited questions for this session.",
                    )
                    submitted_token = st.form_submit_button("Unlock Token", use_container_width=True)
                    if submitted_token:
                        if verify_reviewer_token(token_val):
                            st.session_state["authenticated"] = True
                            st.rerun()
                        else:
                            st.error("Invalid access token. Access denied.")
                render_request_access_link()

        # 6. System Status (collapsible section)
        with st.expander("ℹ️ System Status", expanded=False):
            if is_healthy:
                st.markdown(f'<div style="color: #15803d; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">● Online ({health_msg})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color: #b91c1c; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">● Offline ({health_msg})</div>', unsafe_allow_html=True)
            st.caption(f"Documents loaded: {doc_count}")
            safe_cfg = get_safe_config_summary()
            st.text(f"Environment: {safe_cfg.get('app_env', 'development')}")
            st.text(f"Model: {safe_cfg.get('gemini_model', GEMINI_MODEL)}")
            st.text(f"API Key: {'Configured' if has_gemini_api_key() else 'Missing'}")
            timeout_val = safe_cfg.get("gemini_timeout_seconds", 120.0)
            st.text(f"Timeout: {int(timeout_val) if timeout_val == int(timeout_val) else timeout_val}s")
            st.text(f"Demo Bootstrap: {'Enabled' if is_bootstrap_enabled() else 'Disabled'}")

        # 7. Advanced Configuration (collapsible section)
        with st.expander("⚙️ Advanced Configuration", expanded=False):
            safe_cfg = get_safe_config_summary()
            st.caption(f"Default Top K: {DEFAULT_TOP_K}")
            st.caption(f"Corpus Path: {safe_cfg.get('data_dir', str(PATHS.data_dir))}")

        # 8. Clear Conversation Action at bottom
        # Clears active result, question input, and recent questions history
        if st.button("🗑️ Clear Conversation", key="sidebar_clear_btn", use_container_width=True, help="Clear active result and recent questions history (preserves demo quota and token)"):
            st.session_state["last_result"] = None
            st.session_state["question_input"] = ""
            st.session_state["recent_questions"] = []
            st.rerun()


# -----------------------------------------------------------------------------
# 7. Application Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Personal AI Knowledge Assistant",
        page_icon="🧠",
        layout="wide",
    )

    # Inject static CSS styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "questions_used" not in st.session_state:
        st.session_state["questions_used"] = 0
    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "recent_questions" not in st.session_state:
        st.session_state["recent_questions"] = []
    if "show_more_examples" not in st.session_state:
        st.session_state["show_more_examples"] = False

    # Handle restored session state gracefully if last_result exists but recent_questions has not yet been initialized
    if not st.session_state["recent_questions"] and st.session_state.get("last_result"):
        restored_q = (st.session_state["last_result"] or {}).get("question")
        if restored_q and restored_q.strip():
            st.session_state["recent_questions"] = [restored_q.strip()]

    # 1. Execute optional bootstrap if STREAMLIT_DEMO_BOOTSTRAP is enabled
    bootstrap_ok, bootstrap_err = maybe_bootstrap_index()

    # 2. Check index health
    if bootstrap_ok:
        is_healthy, health_msg = check_index_health()
    else:
        is_healthy = False
        health_msg = f"Bootstrap initialization failed: {bootstrap_err}"

    # Top SaaS Navigation Bar (Phase 1)
    status_class = "online" if is_healthy else "offline"
    status_label = "Online" if is_healthy else "Offline"

    st.markdown(
        f"""
        <div class="saas-top-navbar">
            <div class="saas-brand-identity">
                <div class="saas-brand-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                    </svg>
                </div>
                <div class="saas-brand-text">
                    <div class="saas-brand-title">Personal AI Knowledge Assistant</div>
                    <div class="saas-brand-subtitle">Explore your documents &bull; Get clear answers with sources</div>
                </div>
            </div>
            <div class="saas-navbar-utilities">
                <span class="saas-status-badge {status_class}">
                    <span class="status-dot"></span> {status_label}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Modern SaaS Hero Card (Phase 3)
    st.markdown(
        """
        <div class="saas-hero-card">
            <div class="hero-content">
                <span class="hero-badge-pill">Document-powered AI</span>
                <h2 class="hero-main-title">Ask your knowledge base</h2>
                <p class="hero-main-desc">Get accurate, grounded answers from your documents with clear sources.</p>
            </div>
            <div class="hero-illustration-box">
                <svg width="130" height="85" viewBox="0 0 140 90" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="25" y="10" width="80" height="60" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
                    <line x1="38" y1="24" x2="80" y2="24" stroke="#0d9488" stroke-width="2" stroke-linecap="round"/>
                    <line x1="38" y1="34" x2="92" y2="34" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="38" y1="44" x2="75" y2="44" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
                    <rect x="15" y="20" width="80" height="60" rx="8" fill="#f8fafc" stroke="#0d9488" stroke-width="1.5" fill-opacity="0.9"/>
                    <line x1="28" y1="34" x2="70" y2="34" stroke="#0d9488" stroke-width="2" stroke-linecap="round"/>
                    <line x1="28" y1="44" x2="82" y2="44" stroke="#64748b" stroke-width="1.5" stroke-linecap="round"/>
                    <line x1="28" y1="54" x2="65" y2="54" stroke="#64748b" stroke-width="1.5" stroke-linecap="round"/>
                    <circle cx="95" cy="55" r="16" fill="#f0fdfa" stroke="#0d9488" stroke-width="2.5"/>
                    <line x1="106" y1="66" x2="118" y2="78" stroke="#0d9488" stroke-width="3" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Title & Subtitle Components (preserves test and screen-reader access)
    st.title("Personal AI Knowledge Assistant")
    st.markdown(
        """
        <div class="main-title-wrapper">
            <span class="header-badge">Document-powered AI</span>
            <p class="header-subtitle">Explore your documents. Get clear answers with sources.</p>
            <div class="header-motif-bar">Managed Corpus Retrieval &bull; Grounded Document References</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Allowance computation
    is_authenticated = st.session_state.get("authenticated", False)
    questions_used = st.session_state.get("questions_used", 0)
    remaining = max(0, FREE_QUESTION_LIMIT - questions_used)

    # 4. Display Index Warning Banner if Unhealthy
    if not is_healthy:
        st.error(f"⚠️ **Index Issue Detected:** {health_msg}")

    # 5. Suggested Questions Section
    show_more = st.session_state.get("show_more_examples", False)
    sq_col1, sq_col2 = st.columns([3, 1])
    with sq_col1:
        st.markdown(
            """
            <div class="suggested-section-bar" style="margin-bottom: 0;">
                <span class="suggested-section-title">💡 Suggested Questions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sq_col2:
        st.markdown('<div class="explore-btn-container">', unsafe_allow_html=True)
        toggle_label = "Hide examples ⌃" if show_more else "Explore examples →"
        if st.button(toggle_label, key="btn_explore_examples", use_container_width=True, help="Expand additional example questions"):
            st.session_state["show_more_examples"] = not show_more
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button(
            "📄 Nexus: Model & DB\n\nWhat embedding model and vector database were used?",
            key="ex_nexus",
            use_container_width=True,
            help="Populate Project Nexus question",
        ):
            st.session_state["question_input"] = "What embedding model and vector database were used in Project Nexus?"
            st.rerun()
    with ex_col2:
        if st.button(
            "⚙️ Aurora: Infrastructure\n\nExplain the infrastructure for Project Aurora.",
            key="ex_aurora",
            use_container_width=True,
            help="Populate Project Aurora question",
        ):
            st.session_state["question_input"] = "Explain the infrastructure for Project Aurora."
            st.rerun()
    with ex_col3:
        if st.button(
            "📊 Beacon: Benchmarks\n\nWhat are the key benchmarks in Project Beacon?",
            key="ex_beacon",
            use_container_width=True,
            help="Populate Project Beacon question",
        ):
            st.session_state["question_input"] = "What are the key benchmarks in Project Beacon?"
            st.rerun()

    # Additional expanded example questions
    with st.expander("Explore examples →", expanded=show_more):
        more_col1, more_col2, more_col3 = st.columns(3)
        with more_col1:
            if st.button(
                "👤 Profile: ML Experience\n\nWhat is the candidate's background and experience with ML systems?",
                key="ex_profile",
                use_container_width=True,
                help="Populate candidate profile question",
            ):
                st.session_state["question_input"] = "What is the candidate's background and experience with ML systems?"
                st.rerun()
        with more_col2:
            if st.button(
                "🔄 Architecture Comparison\n\nHow does Project Nexus compare to Project Aurora in architecture?",
                key="ex_compare",
                use_container_width=True,
                help="Populate architecture comparison question",
            ):
                st.session_state["question_input"] = "How does Project Nexus compare to Project Aurora in architecture?"
                st.rerun()
        with more_col3:
            if st.button(
                "📈 Beacon: Target Metrics\n\nWhat evaluation metrics and latency targets were achieved in Beacon?",
                key="ex_beacon_metrics",
                use_container_width=True,
                help="Populate Beacon target metrics question",
            ):
                st.session_state["question_input"] = "What evaluation metrics and latency targets were achieved in Beacon?"
                st.rerun()

    # Layout Containers: Chat Display placed visually ABOVE the Query Form
    chat_container = st.container()
    form_container = st.container()

    with form_container:
        # Counter Badges near the form
        if is_authenticated:
            st.markdown(
                '<div class="allowance-badge unlocked">🔓 Reviewer token active &mdash; unlimited questions</div>',
                unsafe_allow_html=True,
            )
        else:
            if remaining > 0:
                st.markdown(
                    f'<div class="allowance-badge">⚡ {remaining} of {FREE_QUESTION_LIMIT} free questions remaining.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="allowance-badge exhausted">⛔ 0 of {FREE_QUESTION_LIMIT} free questions remaining.</div>',
                    unsafe_allow_html=True,
                )
                st.error("You’ve used your 5 free questions. Contact me for an access token to continue.")
                render_request_access_link()

                with st.expander("🔑 Unlock with Reviewer Access Token", expanded=True):
                    with st.form("exhausted_token_unlock_form", clear_on_submit=True):
                        token_input = st.text_input(
                            "Reviewer Access Token",
                            type="password",
                            placeholder="Enter reviewer access token...",
                        )
                        unlock_submit = st.form_submit_button("Unlock Assistant", type="primary")
                        if unlock_submit:
                            if verify_reviewer_token(token_input):
                                st.session_state["authenticated"] = True
                                st.rerun()
                            else:
                                st.error("Invalid access token. Access denied.")

        # Query Form with Collapsible Retrieval Settings
        with st.form("query_form", clear_on_submit=False):
            question = st.text_area(
                "Question",
                value=st.session_state.get("question_input", ""),
                max_chars=500,
                placeholder="Ask a question about your documents...",
                help="Queries are answered strictly using retrieved evidence. External ungrounded knowledge is rejected.",
            )
            with st.expander("⚙️ Retrieval Settings (top_k chunks)", expanded=False):
                top_k = st.slider(
                    "Number of evidence chunks (top_k)",
                    min_value=1,
                    max_value=10,
                    value=DEFAULT_TOP_K,
                    step=1,
                    help="Number of candidate evidence passages to retrieve from the vector index (default 5).",
                )
            submitted = st.form_submit_button("Ask assistant", type="primary")

    # 6. Handle Explicit Form Submission with Server-Side Quota Enforcement
    if submitted:
        clean_q = (question or "").strip()
        if not clean_q:
            st.warning("Please enter a non-empty question before clicking Ask assistant.")
        elif len(clean_q) > 500:
            st.error("Question exceeds the 500-character limit. Please shorten your question.")
        elif not is_healthy:
            st.error(
                f"Cannot execute query because the File Search store is not ready:\n\n{health_msg}"
            )
        elif not is_authenticated and questions_used >= FREE_QUESTION_LIMIT:
            # Server-side quota guard blocks 6th submission
            st.error("You’ve used your 5 free questions. Contact me for an access token to continue.")
            render_request_access_link()
        else:
            # Record each validated question when the pipeline is invoked, including requests that return an error
            current_recent = st.session_state.get("recent_questions", [])
            if clean_q in current_recent:
                current_recent.remove(clean_q)
            current_recent.insert(0, clean_q)
            st.session_state["recent_questions"] = current_recent[:10]

            with st.spinner("Retrieving evidence and generating grounded answer..."):
                try:
                    result = pipeline.query_pipeline(question=clean_q, top_k=top_k)
                except Exception as e:
                    result = {
                        "question": clean_q,
                        "answer": "",
                        "status": "error",
                        "error": str(e),
                        "citations": [],
                        "retrieved_evidence": [],
                    }
                if not is_authenticated:
                    st.session_state["questions_used"] = questions_used + 1
                st.session_state["last_result"] = result
                st.session_state["question_input"] = ""

    # 7. Render Active Result in chat_container (Visually positioned above the form!)
    active_result = st.session_state.get("last_result")
    if active_result is not None:
        with chat_container:
            q_text = active_result.get("question", "")
            q_escaped = html.escape(q_text)

            # 1. User Question Card
            st.markdown(
                f"""
                <div class="chat-workspace-wrap">
                    <div class="chat-message-row">
                        <div class="chat-avatar-user">U</div>
                        <div class="chat-bubble-user">
                            <div class="chat-bubble-user-text">{q_escaped}</div>
                            <div class="chat-bubble-user-meta">Submitted</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Status Alerts (Maintains full compatibility with widget and status assertions)
            status = active_result.get("status", "unknown")
            if status == "answered":
                st.success("✅ **Status: Answered** — Grounded answer generated from retrieved evidence.")
            elif status == "partial":
                st.warning("⚠️ **Status: Partial** — Grounded answer provided with some requested information missing.")
            elif status == "conflict":
                st.warning("⚠️ **Status: Conflict** — Contradictory evidence detected in the indexed documents.")
            elif status == "abstain":
                st.info("ℹ️ **Status: Abstain** — Insufficient evidence found in the indexed corpus to answer the question.")
            elif status == "error":
                err = active_result.get("error") or "An error occurred during query processing."
                st.error(f"❌ **Status: Error** — {err}")

                err_details = active_result.get("error_details")
                if err_details and isinstance(err_details, dict):
                    with st.expander("Technical Error Diagnostics", expanded=False):
                        if err_details.get("exception_type"):
                            st.markdown(f"**Exception Type:** `{err_details['exception_type']}`")
                        if err_details.get("code") is not None:
                            st.markdown(f"**HTTP / API Code:** `{err_details['code']}`")
                        if err_details.get("model"):
                            st.markdown(f"**Configured Model:** `{err_details['model']}`")
                        if err_details.get("sanitized_reason"):
                            st.markdown(f"**Sanitized Reason:** {err_details['sanitized_reason']}")
            else:
                st.info(f"**Status: {status.title()}**")

            if status == "error":
                st.info("ℹ️ Answer generation and evidence retrieval were aborted due to request failure.")
                with st.expander("Latency Timings", expanded=False):
                    timings = active_result.get("timings", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
                    col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
                    col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")
            else:
                # Assistant Answer Card
                answer_text = active_result.get("answer", "")
                citations = active_result.get("citations", [])
                evidence_list = active_result.get("retrieved_evidence", [])
                num_citations = len(citations)

                with st.container(border=True):
                    # Static decorative assistant header with avatar
                    st.markdown(
                        """<div class="chat-message-row assistant-row"><div class="chat-avatar-assistant"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect width="18" height="14" x="3" y="6" rx="2"></rect><path d="M9 10h.01M15 10h.01"></path><path d="M9 14h6"></path></svg></div><div class="chat-bubble-assistant chat-assistant-header-text"><span class="chat-assistant-name">Knowledge Assistant</span></div></div>""",
                        unsafe_allow_html=True,
                    )

                    # Safe Streamlit Markdown for Answer (never embedded in raw HTML)
                    st.markdown(answer_text)

                    # Native Streamlit Sources expander
                    if citations:
                        with st.expander(f"Sources ({num_citations})", expanded=True):
                            for s_i, cit in enumerate(citations, start=1):
                                c_src = cit.get("source", "Unknown")
                                c_chunk = cit.get("chunk_id", f"chunk_{s_i}")
                                page = cit.get("page")
                                page_text = f" • Page {page}" if page is not None else ""

                                st.write(f"**{s_i}. 📄 `{c_src}`**{page_text}")
                                st.caption(f"Chunk ID: `{c_chunk}`")

                                # Find matching snippet text
                                matched_passage = None
                                for ev in evidence_list:
                                    if ev.get("chunk_id") == cit.get("chunk_id"):
                                        t_raw = ev.get("text", "").strip()
                                        if t_raw:
                                            matched_passage = t_raw
                                        break
                                if matched_passage:
                                    snippet = matched_passage[:240] + ("..." if len(matched_passage) > 240 else "")
                                    st.write(f'"{snippet}"')

                # Missing Information
                missing = active_result.get("missing_information")
                if missing:
                    st.subheader("Missing Information")
                    st.info(missing)

                invalid_citations = active_result.get("invalid_citations", [])
                if invalid_citations:
                    st.warning(f"Unresolved citations detected: {', '.join(invalid_citations)}")

                disclaimer = active_result.get("citation_disclaimer")
                if disclaimer:
                    st.caption(f"*{disclaimer}*")

                # Detailed Retrieved Evidence Chunks expanders
                with st.expander(f"📚 Retrieved Evidence Chunks ({len(evidence_list)})", expanded=False):
                    if not evidence_list:
                        st.caption("No candidate evidence retrieved.")
                    else:
                        for i, ev in enumerate(evidence_list, start=1):
                            chunk_id = ev.get("chunk_id", f"chunk_{i}")
                            source = ev.get("source", "Unknown")
                            distance = ev.get("distance")
                            dist_str = f"{distance:.4f}" if distance is not None else "N/A"
                            page = ev.get("page")
                            page_part = f" | Page {page}" if page is not None else ""

                            expander_label = f"Evidence #{i}: [{chunk_id}]"
                            with st.expander(expander_label, expanded=False):
                                st.markdown(f"**Chunk ID:** `{chunk_id}`")
                                st.markdown(f"**Source Document:** `{source}`{page_part}")
                                if distance is not None:
                                    st.markdown(
                                        f"**Raw Cosine Distance:** `{dist_str}` "
                                        "*(smaller indicates closer similarity; not a probability or confidence score)*"
                                    )
                                else:
                                    st.caption("Relevance scores are not exposed by Gemini File Search.")
                                if ev.get("chunk_index") is not None:
                                    st.markdown(f"**Chunk Index:** `{ev['chunk_index']}`")
                                st.markdown("**Passage Excerpt:**")
                                st.text(ev.get("text", ""))

                # Collapsed Latency Timings
                with st.expander("Latency Timings", expanded=False):
                    timings = active_result.get("timings", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
                    col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
                    col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")

    # 8. Render Sidebar Component (Rendered after form processing so history and quota updates appear immediately)
    render_sidebar(is_healthy, health_msg)


if __name__ == "__main__":
    main()
