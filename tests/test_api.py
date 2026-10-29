"""Unit tests for the FastAPI backend (Phase 5A).

Tests all endpoints using FastAPI's TestClient with a mocked pipeline.
Ensures no paid Gemini API calls or live network operations occur.
"""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.file_search import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    CITATION_VALIDITY_DISCLAIMER,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_pipeline_result() -> dict:
    return {
        "question": "What is Project Aurora?",
        "answer": "Project Aurora is a real-time fraud detection engine deployed on AWS EKS.",
        "status": "answered",
        "missing_information": None,
        "citations": [
            {
                "chunk_id": "chunk_aurora_00",
                "source": "data/documents/project_aurora.md",
                "page": 1,
            }
        ],
        "invalid_citations": [],
        "citation_disclaimer": CITATION_VALIDITY_DISCLAIMER,
        "retrieved_evidence": [
            {
                "chunk_id": "chunk_aurora_00",
                "text": "Project Aurora is deployed on AWS EKS since August 2024.",
                "source": "data/documents/project_aurora.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.22,
            }
        ],
        "timings": {
            "retrieval_ms": 15.2,
            "generation_ms": 320.4,
            "total_ms": 335.6,
        },
        "model": "gemini-2.5-flash",
        "error": None,
    }


# -----------------------------------------------------------------------------
# 1. Health & Liveness Probe
# -----------------------------------------------------------------------------

def test_health_check_returns_ok(client):
    """GET /health must return 200 with service liveness information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "personal-ai-knowledge-assistant"


# -----------------------------------------------------------------------------
# 2. Query Endpoint — Valid Scenarios & All 5 Statuses
# -----------------------------------------------------------------------------

def test_query_endpoint_answered_status(client, sample_pipeline_result):
    """POST /query returns structured 200 response for an answered query."""
    with patch("src.api.query_pipeline", return_value=sample_pipeline_result) as mock_pipe:
        response = client.post(
            "/query",
            json={"question": "What is Project Aurora?", "top_k": 4},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "answered"
        assert "AWS EKS" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "chunk_aurora_00"
        assert len(data["retrieved_evidence"]) == 1
        assert "distance" in data["retrieved_evidence"][0]
        assert data["citation_disclaimer"] == CITATION_VALIDITY_DISCLAIMER
        assert data["timings"]["total_ms"] > 0
        mock_pipe.assert_called_once_with(question="What is Project Aurora?", top_k=4)


def test_query_endpoint_defaults_top_k_to_configured_default(client, sample_pipeline_result):
    """When top_k is omitted in POST /query, it defaults to DEFAULT_TOP_K (5); overrides are preserved."""
    with patch("src.api.query_pipeline", return_value=sample_pipeline_result) as mock_pipe:
        response = client.post(
            "/query",
            json={"question": "What is Project Aurora?"},
            headers={"Authorization": "Bearer test-dev-token"},
        )
        assert response.status_code == 200
        mock_pipe.assert_called_once_with(question="What is Project Aurora?", top_k=5)

    # Test explicit override is preserved
    with patch("src.api.query_pipeline", return_value=sample_pipeline_result) as mock_pipe:
        response = client.post(
            "/query",
            json={"question": "What is Project Aurora?", "top_k": 8},
            headers={"Authorization": "Bearer test-dev-token"},
        )
        assert response.status_code == 200
        mock_pipe.assert_called_once_with(question="What is Project Aurora?", top_k=8)


def test_query_endpoint_abstain_status(client, sample_pipeline_result):
    """POST /query handles abstention when evidence is insufficient."""
    sample_pipeline_result["status"] = "abstain"
    sample_pipeline_result["answer"] = INSUFFICIENT_EVIDENCE_PHRASE
    sample_pipeline_result["citations"] = []
    sample_pipeline_result["missing_information"] = "Salary details are undocumented."

    with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
        response = client.post(
            "/query",
            json={"question": "What is Alex's exact salary?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "abstain"
        assert data["answer"] == INSUFFICIENT_EVIDENCE_PHRASE
        assert data["citations"] == []
        assert data["error"] is None


def test_query_endpoint_partial_status(client, sample_pipeline_result):
    """POST /query preserves partial status and missing_information field."""
    sample_pipeline_result["status"] = "partial"
    sample_pipeline_result["answer"] = "Aurora deployed in August 2024. Beacon is not deployed."
    sample_pipeline_result["missing_information"] = "Beacon benchmarks are undocumented."

    with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
        response = client.post(
            "/query",
            json={"question": "Compare deployment schedules."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert data["missing_information"] == "Beacon benchmarks are undocumented."


def test_query_endpoint_conflict_status(client, sample_pipeline_result):
    """POST /query preserves conflict status."""
    sample_pipeline_result["status"] = "conflict"
    sample_pipeline_result["answer"] = "Sources conflict regarding Nexus deployment status."

    with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
        response = client.post(
            "/query",
            json={"question": "Is Project Nexus deployed?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "conflict"


def test_query_endpoint_pipeline_error_status(client, sample_pipeline_result):
    """POST /query preserves pipeline validation error results cleanly."""
    sample_pipeline_result["status"] = "error"
    sample_pipeline_result["error"] = "Validation error: Model cited invalid chunk ID."

    with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
        response = client.post(
            "/query",
            json={"question": "Tell me about Project Beacon."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Validation error" in data["error"]


# -----------------------------------------------------------------------------
# 3. Input Validation (Pydantic HTTP 422)
# -----------------------------------------------------------------------------

def test_query_endpoint_rejects_empty_question(client):
    """Empty question string must return 422 Unprocessable Entity."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("question" in str(e["loc"]) for e in errors)


def test_query_endpoint_rejects_whitespace_question(client):
    """Whitespace-only question string must return 422."""
    response = client.post("/query", json={"question": "   \n\t  "})
    assert response.status_code == 422


def test_query_endpoint_rejects_missing_question_field(client):
    """Missing question key must return 422."""
    response = client.post("/query", json={"top_k": 4})
    assert response.status_code == 422


def test_query_endpoint_rejects_top_k_less_than_one(client):
    """top_k < 1 must return 422."""
    response = client.post(
        "/query",
        json={"question": "Valid question?", "top_k": 0},
    )
    assert response.status_code == 422


def test_query_endpoint_rejects_top_k_greater_than_ten(client):
    """top_k > 10 must return 422."""
    response = client.post(
        "/query",
        json={"question": "Valid question?", "top_k": 11},
    )
    assert response.status_code == 422


def test_query_endpoint_rejects_invalid_top_k_type(client):
    """Non-integer top_k must return 422."""
    response = client.post(
        "/query",
        json={"question": "Valid question?", "top_k": "four"},
    )
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# 4. Error Safety & Exception Sanitization
# -----------------------------------------------------------------------------

def test_query_endpoint_handles_unhandled_exception_safely(client):
    """Unhandled server errors must return 500 with generic safe message, not leaking secrets."""
    with patch(
        "src.api.query_pipeline",
        side_effect=RuntimeError("Internal crash with sk-secret-1234567890abcdef"),
    ):
        response = client.post(
            "/query",
            json={"question": "Crashing question?"},
        )
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "sk-secret" not in detail
        assert "unexpected error occurred" in detail.lower()


# -----------------------------------------------------------------------------
# 5. CORS Configuration
# -----------------------------------------------------------------------------

def test_cors_allows_local_nextjs_frontend(client):
    """CORS preflight for http://localhost:3000 must be approved."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/query", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unauthorized_origin(client):
    """CORS preflight for unknown origin must not return allow-origin header."""
    headers = {
        "Origin": "http://malicious-external-site.com",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/query", headers=headers)
    assert response.headers.get("access-control-allow-origin") != "http://malicious-external-site.com"


# -----------------------------------------------------------------------------
# 6. Readiness Probe (GET /ready)
# -----------------------------------------------------------------------------

def test_readiness_check_returns_200_when_ready(client):
    """GET /ready returns 200 when the store has documents and the Gemini key is present."""
    with patch("src.api.has_gemini_api_key", return_value=True), \
         patch("src.api.get_store_status", return_value=(True, "ready", 4)):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["document_chunks"] == 4
        assert data["gemini_key_configured"] is True


def test_readiness_check_returns_503_when_no_api_key(client):
    """GET /ready returns 503 when GEMINI_API_KEY is missing."""
    with patch("src.api.has_gemini_api_key", return_value=False):
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert "GEMINI_API_KEY" in data["detail"]


def test_readiness_check_returns_503_when_store_is_empty(client):
    """GET /ready returns 503 when the File Search store has no active documents."""
    with patch("src.api.has_gemini_api_key", return_value=True), \
         patch("src.api.get_store_status", return_value=(False, "store has no active documents", 0)):
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert "no active documents" in data["detail"]


# -----------------------------------------------------------------------------
# 7. Deployment Protections (Phase 5C)
# -----------------------------------------------------------------------------

def test_query_endpoint_rejects_question_exceeding_max_length(client):
    """Questions exceeding 500 characters must return 422 Unprocessable Entity."""
    long_question = "A" * 501
    response = client.post("/query", json={"question": long_question})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("question" in str(e["loc"]) for e in errors)


def test_request_size_limiter_rejects_large_body(client):
    """Requests exceeding 16 KB must return 413 Request Entity Too Large."""
    oversized_padding = "x" * (17 * 1024)
    response = client.post(
        "/query",
        json={"question": "What is Aurora?", "oversized_extra": oversized_padding},
    )
    assert response.status_code == 413
    assert "Request body too large" in response.json()["detail"]


def test_bearer_authentication_when_demo_token_configured(client, sample_pipeline_result):
    """When DEMO_ACCESS_TOKEN is configured, requests must provide valid Bearer token."""
    with patch.dict("os.environ", {"DEMO_ACCESS_TOKEN": "super-secret-token"}):
        # 1. Missing header -> 401
        with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
            res_missing = client.post("/query", json={"question": "What is Aurora?"})
            assert res_missing.status_code == 401
            assert "Missing Authorization header" in res_missing.json()["detail"]

        # 2. Invalid token -> 401
        with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
            res_invalid = client.post(
                "/query",
                json={"question": "What is Aurora?"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert res_invalid.status_code == 401
            assert "Invalid demonstration access token" in res_invalid.json()["detail"]

        # 3. Valid token -> 200
        with patch("src.api.query_pipeline", return_value=sample_pipeline_result):
            res_valid = client.post(
                "/query",
                json={"question": "What is Aurora?"},
                headers={"Authorization": "Bearer super-secret-token"},
            )
            assert res_valid.status_code == 200
            assert res_valid.json()["status"] == "answered"


def test_rate_limiting_enforces_burst_cap(client, sample_pipeline_result):
    """Burst requests from the same client IP exceeding RATE_LIMIT_PER_MINUTE must return 429."""
    from src.api import _ip_request_timestamps
    _ip_request_timestamps.clear()

    with patch.dict("os.environ", {"RATE_LIMIT_PER_MINUTE": "3"}), \
         patch("src.api.RATE_LIMIT_PER_MINUTE", 3), \
         patch("src.api.query_pipeline", return_value=sample_pipeline_result):
        # First 3 queries succeed
        for _ in range(3):
            res = client.post("/query", json={"question": "Valid query?"})
            assert res.status_code == 200

        # 4th query exceeds limit -> 429
        res_blocked = client.post("/query", json={"question": "Valid query?"})
        assert res_blocked.status_code == 429
        assert "Rate limit exceeded" in res_blocked.json()["detail"]
        assert res_blocked.headers.get("retry-after") == "60"

    _ip_request_timestamps.clear()


# -----------------------------------------------------------------------------
# 8. Production Configuration & IP Determination (Phase 5C Revision)
# -----------------------------------------------------------------------------

def test_startup_refuses_when_app_env_is_missing():
    """Startup must refuse when APP_ENV is missing or empty."""
    from src.api import validate_backend_configuration

    with patch.dict("os.environ", {"APP_ENV": ""}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_backend_configuration()
        assert "APP_ENV must be explicitly configured" in str(exc_info.value)
        assert "Set APP_ENV=development" in str(exc_info.value)


def test_startup_refuses_when_app_env_is_mistyped_or_unrecognized():
    """Startup must refuse with actionable error when APP_ENV is mistyped (e.g. staging, prod)."""
    from src.api import validate_backend_configuration

    for mistyped in ["staging", "prod", "dev", "test", "unknown"]:
        with patch.dict("os.environ", {"APP_ENV": mistyped}):
            with pytest.raises(RuntimeError) as exc_info:
                validate_backend_configuration()
            assert "Unrecognized APP_ENV value" in str(exc_info.value)
            assert "Allowed values are strictly" in str(exc_info.value)


def test_production_mode_refuses_start_without_access_token():
    """When APP_ENV=production, backend must refuse to start if DEMO_ACCESS_TOKEN is missing."""
    from src.api import validate_backend_configuration

    with patch.dict("os.environ", {"APP_ENV": "production", "DEMO_ACCESS_TOKEN": ""}):
        with pytest.raises(RuntimeError) as exc_info:
            validate_backend_configuration()
        assert "DEMO_ACCESS_TOKEN must be set when APP_ENV=production" in str(exc_info.value)
        assert "Production authentication must fail closed" in str(exc_info.value)


def test_production_mode_accepts_valid_access_token():
    """When APP_ENV=production and DEMO_ACCESS_TOKEN is configured, validation succeeds."""
    from src.api import validate_backend_configuration

    with patch.dict("os.environ", {"APP_ENV": "production", "DEMO_ACCESS_TOKEN": "prod-secret-token"}):
        env = validate_backend_configuration()
        assert env == "production"


def test_development_mode_preserves_local_convenience():
    """When APP_ENV=development (explicit), missing DEMO_ACCESS_TOKEN is accepted for local pairing."""
    from src.api import validate_backend_configuration

    with patch.dict("os.environ", {"APP_ENV": "development", "DEMO_ACCESS_TOKEN": ""}):
        env = validate_backend_configuration()
        assert env == "development"


def test_client_ip_determination_ignores_untrusted_forwarded_headers():
    """By default (TRUST_PROXY_HEADERS=false), spoofed X-Forwarded-For must be ignored."""
    from src.api import determine_client_ip

    mock_request = MagicMock()
    mock_request.client.host = "192.168.1.50"
    mock_request.headers = {"x-forwarded-for": "203.0.113.195, 10.0.0.1"}

    with patch("src.api.TRUST_PROXY_HEADERS", False):
        ip = determine_client_ip(mock_request)
        assert ip == "192.168.1.50"


def test_client_ip_determination_respects_trusted_proxy_header():
    """When TRUST_PROXY_HEADERS=true, leftmost X-Forwarded-For client IP is extracted."""
    from src.api import determine_client_ip

    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.2"
    mock_request.headers = {"x-forwarded-for": "203.0.113.195, 10.0.0.1"}

    with patch("src.api.TRUST_PROXY_HEADERS", True):
        ip = determine_client_ip(mock_request)
        assert ip == "203.0.113.195"


