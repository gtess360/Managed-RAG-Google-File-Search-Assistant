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


def check_production_access() -> bool:
    """Backwards-compatible helper.

    Replaced by session-based 5-free-questions quota flow. Always returns True
    to permit the 5-question allowance to proceed without an immediate blocking gate.
    """
    return True


# -----------------------------------------------------------------------------
# 5. Modern Static CSS Styling
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Modern AI Portfolio Styling */
:root {
    --navy-900: #0f172a;
    --navy-800: #1e293b;
    --navy-700: #334155;
    --navy-600: #475569;
    --teal-600: #0d9488;
    --teal-700: #0f766e;
    --teal-50: #f0fdfa;
    --teal-100: #ccfbf1;
    --blue-600: #0284c7;
    --border-subtle: #e2e8f0;
    --card-bg: #ffffff;
    --shadow-soft: 0 4px 12px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -2px rgba(15, 23, 42, 0.04);
}

/* Base App Background & Typography */
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    color: var(--navy-800);
}

/* Header Badge */
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

/* Header Subtitle */
.header-subtitle {
    color: var(--navy-600);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: -8px;
    margin-bottom: 12px;
    line-height: 1.5;
}

/* Header Motif Bar */
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

/* Form Container Card */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: var(--shadow-soft);
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
# 6. Application Entrypoint
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

    # Header with Badge, Title, Subtitle, and Decorative Connected Knowledge Motif
    st.markdown('<span class="header-badge">Document-powered AI</span>', unsafe_allow_html=True)
    st.title("Personal AI Knowledge Assistant")
    st.markdown(
        '<p class="header-subtitle">Explore your documents. Get clear answers with sources.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="header-motif-bar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                <circle cx="12" cy="8" r="2"></circle>
                <circle cx="12" cy="14" r="2"></circle>
                <line x1="12" y1="10" x2="12" y2="12"></line>
            </svg>
            <span>Managed Corpus Retrieval &bull; Grounded Document References</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Execute optional bootstrap if STREAMLIT_DEMO_BOOTSTRAP is enabled
    bootstrap_ok, bootstrap_err = maybe_bootstrap_index()

    # 2. Check index health
    if bootstrap_ok:
        is_healthy, health_msg = check_index_health()
    else:
        is_healthy = False
        health_msg = f"Bootstrap initialization failed: {bootstrap_err}"

    # Allowance computation
    is_authenticated = st.session_state.get("authenticated", False)
    questions_used = st.session_state.get("questions_used", 0)
    remaining = max(0, FREE_QUESTION_LIMIT - questions_used)

    # 3. Sidebar: Compact, visually consistent overview
    with st.sidebar:
        st.header("Overview")
        if is_authenticated:
            st.success("Reviewer Token: Active (Unlimited)")
        else:
            st.info(f"{remaining} of {FREE_QUESTION_LIMIT} free questions remaining.")
            st.caption("Session-based demo allowance (resets on new browser session).")

        st.subheader("Reviewer Access")
        if is_authenticated:
            if st.button("Relock Assistant", use_container_width=True):
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

        st.divider()

        # Collapsed Settings
        with st.expander("System & Settings", expanded=False):
            if is_healthy:
                st.caption(f"Index: {health_msg}")
            else:
                st.error(f"Index: {health_msg}")
            safe_cfg = get_safe_config_summary()
            st.text(f"Environment: {safe_cfg.get('app_env', 'development')}")
            st.text(f"Model: {safe_cfg.get('gemini_model', GEMINI_MODEL)}")
            st.text(f"API Key: {'Configured' if has_gemini_api_key() else 'Missing'}")
            timeout_val = safe_cfg.get("gemini_timeout_seconds", 120.0)
            st.text(f"Timeout: {int(timeout_val) if timeout_val == int(timeout_val) else timeout_val}s")
            st.text(f"Demo Bootstrap: {'Enabled' if is_bootstrap_enabled() else 'Disabled'}")

        if st.button("Clear Saved Result", use_container_width=True):
            st.session_state["last_result"] = None
            st.rerun()

    # 4. Display Index Warning Banner if Unhealthy
    if not is_healthy:
        st.error(f"⚠️ **Index Issue Detected:** {health_msg}")

    # 5. Question Workspace Card
    # Counter Badge near the form
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
            st.caption(f"{remaining} of {FREE_QUESTION_LIMIT} free questions remaining. (Session-based demo allowance)")
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

    # Compact Example Questions that populate input without submitting
    st.markdown("##### 💡 Example Questions")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("Nexus: Model & DB", key="ex_nexus", use_container_width=True, help="Populate question"):
            st.session_state["question_input"] = "What embedding model and vector database were used in Project Nexus?"
            st.rerun()
    with ex_col2:
        if st.button("Aurora: Infrastructure", key="ex_aurora", use_container_width=True, help="Populate question"):
            st.session_state["question_input"] = "What deployment environment and target cluster were used for Project Aurora?"
            st.rerun()
    with ex_col3:
        if st.button("Beacon: Benchmarks", key="ex_beacon", use_container_width=True, help="Populate question"):
            st.session_state["question_input"] = "What throughput and latency benchmarks were recorded for Project Beacon?"
            st.rerun()

    # Query Form
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            value=st.session_state.get("question_input", ""),
            max_chars=500,
            placeholder="e.g. What embedding model and vector database were used in Project Nexus?",
            help="Queries are answered strictly using retrieved evidence. External ungrounded knowledge is rejected.",
        )
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
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                result = pipeline.query_pipeline(question=clean_q, top_k=top_k)
                if not is_authenticated:
                    st.session_state["questions_used"] = questions_used + 1
                st.session_state["last_result"] = result
                st.session_state["question_input"] = clean_q

    # 7. Render Last Result from Session State
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()

        # Status Display
        status = result.get("status", "unknown")
        if status == "answered":
            st.success("✅ **Status: Answered** — Grounded answer generated from retrieved evidence.")
        elif status == "partial":
            st.warning("⚠️ **Status: Partial** — Grounded answer provided with some requested information missing.")
        elif status == "conflict":
            st.warning("⚠️ **Status: Conflict** — Contradictory evidence detected in the indexed documents.")
        elif status == "abstain":
            st.info("ℹ️ **Status: Abstain** — Insufficient evidence found in the indexed corpus to answer the question.")
        elif status == "error":
            err = result.get("error") or "An error occurred during query processing."
            st.error(f"❌ **Status: Error** — {err}")

            err_details = result.get("error_details")
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
            # On error, suppress Answer, Citations, and Evidence sections completely.
            st.info("ℹ️ Answer generation and evidence retrieval were aborted due to request failure.")

            # Collapsed Latency Timings
            with st.expander("Latency Timings", expanded=False):
                timings = result.get("timings", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
                col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
                col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")
        else:
            # Answer (Primary visual focus, readable typography)
            st.subheader("Answer")
            answer_text = result.get("answer", "")
            if answer_text:
                st.write(answer_text)
            else:
                st.caption("No answer text provided.")

            # Missing Information
            missing = result.get("missing_information")
            if missing:
                st.subheader("Missing Information")
                st.info(missing)

            # Citations (Source references beneath the answer)
            st.subheader("Citations")
            citations = result.get("citations", [])
            if citations:
                for cit in citations:
                    page_part = f", Page {cit['page']}" if cit.get("page") is not None else ""
                    st.markdown(f"- **`{cit.get('chunk_id', '')}`** → Source: `{cit.get('source', '')}`{page_part}")
            else:
                st.caption("No citations cited in this answer.")

            invalid_citations = result.get("invalid_citations", [])
            if invalid_citations:
                st.warning(f"Unresolved citations detected: {', '.join(invalid_citations)}")

            disclaimer = result.get("citation_disclaimer")
            if disclaimer:
                st.caption(f"*{disclaimer}*")

            # Expandable Supporting Evidence
            st.subheader("Retrieved Evidence Chunks")
            evidence_list = result.get("retrieved_evidence", [])
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
                timings = result.get("timings", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieval Latency", f"{timings.get('retrieval_ms', 0.0):.1f} ms")
                col2.metric("Generation Latency", f"{timings.get('generation_ms', 0.0):.1f} ms")
                col3.metric("Total Latency", f"{timings.get('total_ms', 0.0):.1f} ms")


if __name__ == "__main__":
    main()
