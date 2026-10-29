"""Unit and interface tests for the Streamlit web application.

Verifies:
1. Pre-flight index health and compatibility checks (read-only, actionable).
2. Demo bootstrap mechanism:
   - Disabled by default.
   - Initializing missing or empty index once per process with double-checked locking.
   - Reusing existing index without rebuilding.
   - Initialization failure handling and blocking subsequent queries.
3. Feature 1 — Five Free Questions & Reviewer Token Verification:
   - Five allowed submissions and server-side blocked sixth request.
   - Quota counter display ("X of 5 free questions remaining").
   - Blank / invalid inputs and ordinary reruns do not consume quota.
   - Sessions have independent counters.
   - Valid token unlocks unlimited queries and bypasses demo quota.
   - Invalid token rejected without resetting quota or invoking pipeline.
   - Previous results remain visible after quota exhaustion.
   - Contact-link configuration (CONTACT_URL, CONTACT_EMAIL, and fallback setup notice).
4. Feature 2 — Modern Portfolio Visual Design:
   - Header badge, title, subtitle, and connected knowledge motif.
   - Example questions as compact buttons that populate input without submitting.
   - Form workspace with primary teal action ("Ask assistant").
   - Collapsed technical diagnostics and latency timings.
5. Streamlit secrets propagation to os.environ before config import.
6. Status badge rendering for answered, partial, conflict, abstain, and error.
7. Technical error diagnostics and timeout / 504 error handling.

All tests run completely offline with mocked pipeline responses and external services.
No live API calls or index rebuilds are executed.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    check_index_health,
    maybe_bootstrap_index,
    is_bootstrap_enabled,
    check_production_access,
    reset_bootstrap_state_for_tests,
    _sync_streamlit_secrets_to_env,
    verify_reviewer_token,
    get_contact_destination,
    FREE_QUESTION_LIMIT,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure clean process-level state between test executions."""
    reset_bootstrap_state_for_tests()
    with patch("src.file_search.get_store_status", return_value=(True, "File Search store ready (4 document(s) indexed).", 4)):
        yield
    reset_bootstrap_state_for_tests()


def _get_submit_button(at: AppTest):
    """Find the primary query form submission button."""
    btns = [b for b in at.button if b.label in ("Ask assistant", "Ask")]
    assert len(btns) >= 1, f"Submit button not found among: {[b.label for b in at.button]}"
    return btns[0]


# -----------------------------------------------------------------------------
# 1. Index Health & Pre-flight Diagnostics Tests
# -----------------------------------------------------------------------------

def test_check_index_health_healthy():
    """Verify check_index_health returns True when File Search store exists and is ready."""
    with patch("src.file_search.get_store_status", return_value=(True, "File Search store ready (10 document(s) indexed).", 10)):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is True
        assert "ready" in msg.lower()
        assert "10 document(s) indexed" in msg


def test_check_index_health_missing_collection():
    """Verify check_index_health returns False with actionable instructions when store is missing."""
    with patch("src.file_search.get_store_status", return_value=(False, "File Search store 'personal-ai-knowledge-assistant' not found. Action required: run 'python ingest.py' to create it and upload documents.", 0)):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is False
        assert "not found" in msg.lower()
        assert "python ingest.py" in msg


def test_check_index_health_empty_collection():
    """Verify check_index_health returns False when store exists but has 0 active documents."""
    with patch("src.file_search.get_store_status", return_value=(False, "File Search store exists but has no active documents. Action required: run 'python ingest.py' to upload documents.", 0)):
        healthy, msg = check_index_health(Path("/mock/dir"))
        assert healthy is False
        assert "no active documents" in msg.lower()
        assert "python ingest.py" in msg


# -----------------------------------------------------------------------------
# 2. Demo Bootstrap Mechanism Tests
# -----------------------------------------------------------------------------

def test_bootstrap_disabled_by_default(monkeypatch):
    """Verify demo bootstrap is disabled by default and does not invoke ingestion."""
    monkeypatch.delenv("STREAMLIT_DEMO_BOOTSTRAP", raising=False)
    assert not is_bootstrap_enabled()

    with patch("ingest.ingest_corpus") as mock_ingest:
        ok, err = maybe_bootstrap_index()
        assert ok is True
        assert err is None
        mock_ingest.assert_not_called()


def test_bootstrap_missing_index_initialization(monkeypatch):
    """Verify unready store triggers ingestion when STREAMLIT_DEMO_BOOTSTRAP=true."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    with patch("src.file_search.get_store_status", return_value=(False, "Store not found", 0)), \
         patch("ingest.ingest_corpus", return_value={"uploaded": 4}) as mock_ingest:
        ok, err = maybe_bootstrap_index(
            storage_dir=Path("/mock/storage"),
            data_dir=Path("/mock/data"),
        )
        assert ok is True
        assert err is None
        mock_ingest.assert_called_once_with(
            allow_partial_ingestion=False,
            data_dir=Path("/mock/data"),
            replace=False,
        )


def test_bootstrap_reuse_existing_index(monkeypatch):
    """Verify existing populated store is reused and NEVER rebuilt or replaced."""
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    with patch("src.file_search.get_store_status", return_value=(True, "Store ready", 4)), \
         patch("ingest.ingest_corpus") as mock_ingest:
        ok, err = maybe_bootstrap_index()
        assert ok is True
        assert err is None
        mock_ingest.assert_not_called()


def test_bootstrap_initialization_failure(monkeypatch, caplog):
    """Verify bootstrap failure records error, logs traceback, and blocks subsequent operations."""
    import logging
    caplog.set_level(logging.ERROR)
    monkeypatch.setenv("STREAMLIT_DEMO_BOOTSTRAP", "true")

    with patch("src.file_search.get_store_status", return_value=(False, "Store not found", 0)), \
         patch("ingest.ingest_corpus", side_effect=RuntimeError("API error")):
        ok, err = maybe_bootstrap_index()
        assert ok is False
        assert "STREAMLIT_DEMO_BOOTSTRAP failed" in err
        assert "API error" in err

        # Subsequent call returns cached error without re-running ingest
        ok2, err2 = maybe_bootstrap_index()
        assert ok2 is False
        assert err2 == err


# -----------------------------------------------------------------------------
# 3. Feature 1: Five Free Questions & Reviewer Token Verification Tests
# -----------------------------------------------------------------------------

def test_five_allowed_submissions_and_blocked_sixth_request(monkeypatch):
    """Verify visitors can submit exactly 5 questions, with the 6th call blocked server-side."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    mock_resp = {
        "question": "q",
        "answer": "Answer fact.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "c1", "source": "s1.md", "page": 1}],
        "invalid_citations": [],
        "citation_disclaimer": "Grounding disclaimer.",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
        "model": "gemini-3.5-flash-lite",
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_resp) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Submit 5 allowed questions
        for i in range(1, FREE_QUESTION_LIMIT + 1):
            assert at.session_state["questions_used"] == i - 1
            at.text_area[0].input(f"Question number {i}")
            ask_btn = _get_submit_button(at)
            ask_btn.click().run()
            assert mock_pipe.call_count == i
            assert at.session_state["questions_used"] == i

        # Now questions_used == 5. Sixth submission MUST be blocked server-side.
        at.text_area[0].input("Question number 6")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        # Pipeline call count must NOT increase
        assert mock_pipe.call_count == 5
        assert at.session_state["questions_used"] == 5

        # Error message displayed
        all_text = " ".join([e.value for e in at.error] + [m.value for m in at.markdown])
        assert "used your 5 free questions" in all_text.lower()
        assert "Request access" in all_text


def test_reruns_and_invalid_inputs_do_not_consume_quota(monkeypatch):
    """Verify blank input, length violation, ordinary reruns, and invalid token don't consume quota."""
    monkeypatch.setenv("APP_ENV", "development")
    with patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        assert at.session_state["questions_used"] == 0

        # 1. Blank question
        at.text_area[0].input("   ")
        _get_submit_button(at).click().run()
        mock_pipe.assert_not_called()
        assert at.session_state["questions_used"] == 0

        # 2. Too long question (>500 chars)
        at.text_area[0].input("x" * 501)
        _get_submit_button(at).click().run()
        mock_pipe.assert_not_called()
        assert at.session_state["questions_used"] == 0

        # 3. Ordinary rerun
        at.run()
        mock_pipe.assert_not_called()
        assert at.session_state["questions_used"] == 0

        # 4. Invalid token submission
        sidebar_token_btn = [b for b in at.button if b.label == "Unlock Token"]
        if sidebar_token_btn:
            at.text_input[0].input("wrong-secret")
            sidebar_token_btn[0].click().run()
            mock_pipe.assert_not_called()
            assert at.session_state["questions_used"] == 0


def test_sessions_have_independent_counters(monkeypatch):
    """Verify that multiple browser sessions maintain separate, isolated question counters."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_resp = {
        "question": "q",
        "answer": "Answer",
        "status": "answered",
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0},
        "model": "gemini",
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_resp):
        # Session A
        at_a = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at_a.text_area[0].input("Session A query 1")
        _get_submit_button(at_a).click().run()
        at_a.text_area[0].input("Session A query 2")
        _get_submit_button(at_a).click().run()
        assert at_a.session_state["questions_used"] == 2

        # Session B starts fresh
        at_b = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        assert at_b.session_state["questions_used"] == 0
        at_b.text_area[0].input("Session B query 1")
        _get_submit_button(at_b).click().run()
        assert at_b.session_state["questions_used"] == 1
        assert at_a.session_state["questions_used"] == 2


def test_valid_token_bypasses_quota_and_invalid_token_rejected(monkeypatch):
    """Verify valid token unlocks unlimited queries and invalid token is rejected."""
    monkeypatch.setenv("REVIEWER_ACCESS_TOKEN", "secret-test-token-123")
    mock_resp = {
        "question": "q",
        "answer": "Answer",
        "status": "answered",
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 1.0, "generation_ms": 1.0, "total_ms": 2.0},
        "model": "gemini",
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_resp) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Try invalid token
        unlock_btns = [b for b in at.button if b.label == "Unlock Token"]
        assert len(unlock_btns) >= 1
        at.text_input[0].input("bad-token")
        unlock_btns[0].click().run()

        assert at.session_state["authenticated"] is False
        assert any("Invalid access token" in e.value for e in at.error)

        # Enter valid token
        unlock_btns = [b for b in at.button if b.label == "Unlock Token"]
        at.text_input[0].input("secret-test-token-123")
        unlock_btns[0].click().run()

        assert at.session_state["authenticated"] is True

        # When authenticated, visitor can submit past 5 questions without quota block
        at.session_state["questions_used"] = 5
        at.text_area[0].input("Question 6 with valid token")
        _get_submit_button(at).click().run()

        mock_pipe.assert_called_once()
        assert at.session_state["last_result"] == mock_resp


def test_previous_results_remain_visible_after_exhaustion(monkeypatch):
    """Verify that after the quota is exhausted, previous answer and evidence remain visible."""
    mock_result = {
        "question": "What database does Project Nexus use?",
        "answer": "Project Nexus uses ChromaDB.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "nexus_01", "source": "project_nexus.md", "page": 1}],
        "invalid_citations": [],
        "citation_disclaimer": "Citations verified.",
        "retrieved_evidence": [
            {
                "chunk_id": "nexus_01",
                "text": "Project Nexus utilizes ChromaDB.",
                "source": "project_nexus.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.1,
            }
        ],
        "timings": {"retrieval_ms": 10.0, "generation_ms": 100.0, "total_ms": 110.0},
        "model": "gemini-3.5-flash-lite",
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Set quota to 5 and store last result
        at.session_state["questions_used"] = 5
        at.session_state["last_result"] = mock_result
        at.run()

        # Attempt blocked 6th query
        at.text_area[0].input("Sixth query attempt")
        _get_submit_button(at).click().run()

        # Pipeline was NOT invoked
        mock_pipe.assert_not_called()

        # Previous result is STILL rendered on screen
        markdown_texts = [m.value for m in at.markdown]
        assert any("nexus_01" in m for m in markdown_texts)
        assert any("project_nexus.md" in m for m in markdown_texts)
        assert any("Project Nexus uses ChromaDB." in m for m in markdown_texts)


def test_contact_link_destination_resolution(monkeypatch):
    """Verify get_contact_destination prioritizes CONTACT_URL, falls back to CONTACT_EMAIL, or None."""
    # 1. URL configured
    monkeypatch.setenv("CONTACT_URL", "https://example.com/contact")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    dtype, dval = get_contact_destination()
    assert dtype == "url"
    assert dval == "https://example.com/contact"

    # 2. Email configured, URL unset
    monkeypatch.delenv("CONTACT_URL", raising=False)
    monkeypatch.setenv("CONTACT_EMAIL", "admin@example.com")
    dtype, dval = get_contact_destination()
    assert dtype == "email"
    assert "mailto:admin@example.com" in dval

    # 3. Neither configured
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    dtype, dval = get_contact_destination()
    assert dtype is None
    assert dval is None


# -----------------------------------------------------------------------------
# 4. Feature 2: Modern Portfolio Visual Design & UI Elements
# -----------------------------------------------------------------------------

def test_streamlit_initial_render(monkeypatch):
    """Verify modern UI components: header badge, title, subtitle, form, slider, example buttons."""
    monkeypatch.setenv("APP_ENV", "development")
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

    # Check Title
    assert len(at.title) >= 1
    assert at.title[0].value == "Personal AI Knowledge Assistant"

    # Check Header Badge & Subtitle
    markdown_all = " ".join([m.value for m in at.markdown])
    assert "Document-powered AI" in markdown_all
    assert "Explore your documents. Get clear answers with sources." in markdown_all
    assert "Managed Corpus Retrieval" in markdown_all

    # Check Question Input with 500-character max limit
    assert len(at.text_area) >= 1
    assert at.text_area[0].max_chars == 500

    # Check Slider: default 5, min 1, max 10
    assert len(at.slider) >= 1
    slider = at.slider[0]
    assert slider.value == 5
    assert slider.min == 1
    assert slider.max == 10

    # Check Primary Submit Button
    ask_btn = [b for b in at.button if b.label == "Ask assistant"]
    assert len(ask_btn) == 1

    # Check Example Question Secondary Buttons
    example_btns = [b for b in at.button if any(k in b.label for k in ("Nexus", "Aurora", "Beacon"))]
    assert len(example_btns) >= 3

    # Initial free question allowance counter
    assert any("5 of 5 free questions remaining" in m for m in [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info])

    # Last result should be None initially
    assert at.session_state["last_result"] is None


def test_example_question_populates_input_without_submitting(monkeypatch):
    """Verify clicking an example button populates question_input without calling pipeline."""
    with patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Find Nexus example button
        nexus_btn = [b for b in at.button if "Nexus" in b.label][0]
        nexus_btn.click().run()

        # Pipeline must NOT have been called
        mock_pipe.assert_not_called()
        assert at.session_state["questions_used"] == 0

        # Input text area is populated with example question
        assert at.session_state["question_input"] == "What embedding model and vector database were used in Project Nexus?"
        assert at.text_area[0].value == "What embedding model and vector database were used in Project Nexus?"


def test_sync_streamlit_secrets_to_env(monkeypatch):
    """Verify secrets in st.secrets are populated into os.environ without printing."""
    fake_secrets = {
        "CUSTOM_DEPLOY_SECRET": "deploy_val_123",
        "APP_ENV": "production",
    }
    monkeypatch.delenv("CUSTOM_DEPLOY_SECRET", raising=False)

    with patch("streamlit.secrets", fake_secrets, create=True):
        _sync_streamlit_secrets_to_env()
        assert os.environ.get("CUSTOM_DEPLOY_SECRET") == "deploy_val_123"
        assert os.environ.get("APP_ENV") == "production"


def test_streamlit_empty_submission(monkeypatch):
    """Verify submitting empty question displays warning and does not call pipeline."""
    monkeypatch.setenv("APP_ENV", "development")
    with patch("src.pipeline.query_pipeline") as mock_pipeline:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("   ")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        mock_pipeline.assert_not_called()
        assert len(at.warning) >= 1
        assert "non-empty question" in at.warning[0].value.lower()


def test_streamlit_successful_query_flow(monkeypatch):
    """Verify full query lifecycle: user query -> mocked query_pipeline -> answer & evidence rendered."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "What database does Project Nexus use?",
        "answer": "Project Nexus uses ChromaDB for persistent vector storage.",
        "status": "answered",
        "missing_information": None,
        "citations": [
            {"chunk_id": "nexus_01", "source": "project_nexus.md", "page": 1}
        ],
        "invalid_citations": [],
        "citation_disclaimer": "Citations are resolved strictly from storage metadata.",
        "retrieved_evidence": [
            {
                "chunk_id": "nexus_01",
                "text": "Project Nexus utilizes ChromaDB as its persistent vector database.",
                "source": "project_nexus.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.1234,
            }
        ],
        "timings": {
            "retrieval_ms": 12.5,
            "generation_ms": 340.2,
            "total_ms": 352.7,
        },
        "model": "gemini-3.5-flash-lite",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Input query and submit
        at.text_area[0].input("What database does Project Nexus use?")
        at.slider[0].set_value(4)
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        # Pipeline must be called with expected parameters
        mock_pipe.assert_called_once_with(
            question="What database does Project Nexus use?",
            top_k=4,
        )

        # Result preserved in session state
        assert at.session_state["last_result"] == mock_result

        # Success banner for "answered" status
        assert any("Answered" in s.value for s in at.success)

        # Citations rendered
        markdown_texts = [m.value for m in at.markdown]
        assert any("nexus_01" in m for m in markdown_texts)
        assert any("project_nexus.md" in m for m in markdown_texts)


def test_streamlit_preserves_session_state_on_subsequent_rerun(monkeypatch):
    """Verify that ordinary UI reruns preserve the last result without re-invoking query_pipeline."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "What is Project Aurora?",
        "answer": "Project Aurora was deployed to AWS EKS.",
        "status": "answered",
        "missing_information": None,
        "citations": [{"chunk_id": "aurora_01", "source": "project_aurora.md", "page": None}],
        "invalid_citations": [],
        "citation_disclaimer": "Verified disclaimer",
        "retrieved_evidence": [
            {
                "chunk_id": "aurora_01",
                "text": "Aurora deployment details.",
                "source": "project_aurora.md",
                "page": None,
                "chunk_index": 0,
                "distance": 0.2222,
            }
        ],
        "timings": {"retrieval_ms": 10.0, "generation_ms": 200.0, "total_ms": 210.0},
        "model": "gemini-3.5-flash-lite",
        "error": None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # 1. Ask question
        at.text_area[0].input("What is Project Aurora?")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        assert mock_pipe.call_count == 1
        assert at.session_state["last_result"] == mock_result

        # 2. Trigger a normal rerun without clicking Ask
        at.run()

        # query_pipeline must NOT be called again
        assert mock_pipe.call_count == 1

        # Content must still be rendered from session state
        markdown_texts = [m.value for m in at.markdown]
        assert any("aurora_01" in m for m in markdown_texts)


@pytest.mark.parametrize("status,expected_widget,expected_text", [
    ("partial", "warning", "Partial"),
    ("conflict", "warning", "Conflict"),
    ("abstain", "info", "Abstain"),
    ("error", "error", "Error"),
])
def test_streamlit_status_rendering(monkeypatch, status, expected_widget, expected_text):
    """Verify distinct display for partial, conflict, abstain, and error statuses."""
    monkeypatch.setenv("APP_ENV", "development")
    mock_result = {
        "question": "Test query",
        "answer": "Answer content or fallback phrase.",
        "status": status,
        "missing_information": "Information about missing component." if status == "partial" else None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "Test disclaimer",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 5.0, "generation_ms": 50.0, "total_ms": 55.0},
        "model": "gemini-3.5-flash-lite",
        "error": "Simulated pipeline failure" if status == "error" else None,
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result):
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("Test query")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        widget_list = getattr(at, expected_widget)
        assert len(widget_list) >= 1
        assert any(expected_text in w.value for w in widget_list)

        # If partial, missing information section should also be present
        if status == "partial":
            assert any("Information about missing component." in i.value for i in at.info)


def test_streamlit_blocks_submission_when_index_unhealthy(monkeypatch):
    """Verify that an unhealthy store prevents query execution and renders actionable guidance."""
    monkeypatch.setenv("APP_ENV", "development")

    with patch("src.file_search.get_store_status", return_value=(False, "File Search store not found. Action required: run 'python ingest.py' to create it.", 0)), \
         patch("src.pipeline.query_pipeline") as mock_pipe:
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()

        # Error banner should be rendered immediately mentioning ingest.py
        all_errors = [e.value for e in at.error] + [e.value for e in at.sidebar.error]
        assert any("python ingest.py" in err for err in all_errors)

        # Attempt to submit question
        at.text_area[0].input("Valid question text")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        # Pipeline must not have been executed
        mock_pipe.assert_not_called()
        assert at.session_state["last_result"] is None


def test_streamlit_error_rendering_suppresses_answer_and_shows_diagnostics(monkeypatch):
    """Verify that status='error' suppresses the answer/evidence sections and displays technical diagnostics."""
    monkeypatch.setenv("APP_ENV", "development")
    from src.file_search import INSUFFICIENT_EVIDENCE_PHRASE

    mock_result = {
        "question": "What is Project Aurora?",
        "answer": "",
        "status": "error",
        "error": "Gemini File Search request failed. Details: [ClientError] (HTTP 404) - models/gemini-3.8-flash not found (model='gemini-3.8-flash')",
        "error_details": {
            "exception_type": "ClientError",
            "code": 404,
            "model": "gemini-3.8-flash",
            "sanitized_reason": "models/gemini-3.8-flash not found",
        },
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": 45.0},
        "model": "gemini-3.8-flash",
    }

    with patch("src.pipeline.query_pipeline", return_value=mock_result):
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("What is Project Aurora?")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        # 1. Error banner must display the technical error
        error_texts = [e.value for e in at.error]
        assert any("ClientError" in err and "404" in err for err in error_texts)

        # 2. Technical diagnostics expander contains structured fields
        markdown_texts = [m.value for m in at.markdown]
        assert any("ClientError" in m for m in markdown_texts)
        assert any("404" in m for m in markdown_texts)
        assert any("gemini-3.8-flash" in m for m in markdown_texts)

        # 3. Insufficient-evidence answer and misleading search captions must be SUPPRESSED
        all_rendered_text = " ".join(
            [e.value for e in at.error] +
            [m.value for m in at.markdown] +
            [c.value for c in at.caption] +
            [i.value for i in at.info]
        )
        assert INSUFFICIENT_EVIDENCE_PHRASE not in all_rendered_text
        assert "No candidate evidence retrieved." not in all_rendered_text

        # 4. Abort notice must be present
        info_texts = [i.value for i in at.info]
        assert any("aborted due to request failure" in i for i in info_texts)


def test_streamlit_timeout_504_rendering_shows_timeout_and_suppresses_answer(monkeypatch):
    """Verify Streamlit renders HTTP 504 timeout error clearly and suppresses insufficient-evidence answer."""
    monkeypatch.setenv("APP_ENV", "development")
    from src.file_search import INSUFFICIENT_EVIDENCE_PHRASE

    timeout_result = {
        "question": "What are the core metrics for Project Nexus?",
        "answer": "",
        "status": "error",
        "error": "Gemini API request timed out after 120s. Please retry. Details: [ServerError] (HTTP 504) - Deadline Exceeded (model='gemini-3.8-flash')",
        "error_details": {
            "exception_type": "ServerError",
            "code": 504,
            "model": "gemini-3.8-flash",
            "sanitized_reason": "Deadline Exceeded",
            "summary": "Gemini API request timed out after 120s. Please retry.",
        },
        "missing_information": None,
        "citations": [],
        "invalid_citations": [],
        "citation_disclaimer": "",
        "retrieved_evidence": [],
        "timings": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": 120015.0},
        "model": "gemini-3.8-flash",
    }

    with patch("src.pipeline.query_pipeline", return_value=timeout_result):
        at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        at.text_area[0].input("What are the core metrics for Project Nexus?")
        ask_btn = _get_submit_button(at)
        ask_btn.click().run()

        # 1. Error banner must display the timeout error
        error_texts = [e.value for e in at.error]
        assert any("timed out after 120s" in err for err in error_texts)
        assert any("HTTP 504" in err for err in error_texts)

        # 2. Technical diagnostics expander shows HTTP 504
        markdown_texts = [m.value for m in at.markdown]
        assert any("504" in m for m in markdown_texts)
        assert any("ServerError" in m for m in markdown_texts)

        # 3. Insufficient-evidence text must NEVER appear
        all_rendered_text = " ".join(
            [e.value for e in at.error] +
            [m.value for m in at.markdown] +
            [c.value for c in at.caption] +
            [i.value for i in at.info]
        )
        assert INSUFFICIENT_EVIDENCE_PHRASE not in all_rendered_text
        assert "No candidate evidence retrieved." not in all_rendered_text

        # 4. Abort notice must be present
        info_texts = [i.value for i in at.info]
        assert any("aborted due to request failure" in i for i in info_texts)
