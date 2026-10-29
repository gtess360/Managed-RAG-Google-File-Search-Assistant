"""FastAPI backend application for the Personal AI Knowledge Assistant.

Exposes the core RAG pipeline via REST endpoints for consumption by the Next.js frontend:
- GET /health: Liveness health check.
- GET /ready: Readiness check verifying persistent index and configuration.
- POST /query: Executes the evidence-grounded RAG pipeline.

Design & Operational Principles:
1. Reuses `src.pipeline.query_pipeline`; contains no duplicated chunking, embedding, or generation logic.
2. Runs the synchronous CPU embedding and network generation pipeline in a worker threadpool
   so the asynchronous event loop is never blocked.
3. Caches and reuses the local embedding model across all requests.
4. Does NOT ingest, rebuild, or alter the index at startup or during queries.
5. Does NOT expose any ingestion or document-upload endpoints.
6. Keeps OpenAI credentials strictly confined to the backend server environment.
7. Enforces exact CORS origins configured via ALLOWED_ORIGINS, rejecting wildcards.
   Note: CORS is a browser cross-origin policy, NOT an authentication or authorization mechanism.
8. Binds locally to 127.0.0.1 by default, or configurable HOST/PORT in containers.
9. Distinguishes basic service liveness (/health) from index/model readiness (/ready).
10. Deployment protections (Phase 5C):
    - Configurable Bearer token authentication via DEMO_ACCESS_TOKEN.
    - Bounded request body size (<= 16 KB).
    - Bounded query character length (<= 500 characters).
    - In-memory sliding window rate limiting (per client IP).
    - Concurrency semaphore bounding simultaneous pipeline executions.
    - Safe error scrubbing preventing leakage of internal traces or paths.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from src.pipeline import query_pipeline
from src.file_search import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    CITATION_VALIDITY_DISCLAIMER,
    get_store_status,
    sanitize_error_message,
)
from src.config import GEMINI_MODEL, has_gemini_api_key, APP_ENV, DEFAULT_TOP_K


# -----------------------------------------------------------------------------
# Deployment Configuration & Protections
# -----------------------------------------------------------------------------

# Maximum request body size in bytes (16 KB)
MAX_REQUEST_BODY_BYTES = 16 * 1024

# Maximum concurrent queries handled simultaneously
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "3"))

# Demonstration rate limit: queries per minute per client IP
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "15"))

# Allowed frontend origins (exact matches, rejecting wildcards)
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _raw_origins.split(",")
    if origin.strip() and origin.strip() != "*"
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Reverse proxy header trust setting
# By default, do NOT trust arbitrary forwarded headers from the network to prevent spoofing.
# Set TRUST_PROXY_HEADERS=true only when deployed behind a known proxy that strips/replaces X-Forwarded-For.
TRUST_PROXY_HEADERS: bool = (
    os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in ("true", "1", "yes")
)


VALID_APP_ENVS = frozenset({"development", "production"})


def validate_backend_configuration() -> str:
    """Validate backend startup configuration and fail closed on missing or invalid settings.

    Requirements:
    1. APP_ENV must be explicitly configured as 'development' or 'production'.
       Missing, blank, or unrecognized values are rejected with an actionable startup error.
    2. When APP_ENV=production, DEMO_ACCESS_TOKEN must be non-empty. Refuses to start without it.
    3. When APP_ENV=development, convenient open local access is permitted if DEMO_ACCESS_TOKEN is unset.

    Returns:
        The validated app environment ('development' or 'production').

    Raises:
        RuntimeError: If APP_ENV is missing, unrecognized, or production token is missing.
    """
    raw_env = os.getenv("APP_ENV")
    if raw_env is None or not raw_env.strip():
        raise RuntimeError(
            "Backend startup failed: APP_ENV must be explicitly configured.\n"
            "Allowed values are strictly 'development' or 'production'.\n"
            "Action required: Set APP_ENV=development in your root .env file or environment for local development."
        )
    env = raw_env.strip().lower()
    if env not in VALID_APP_ENVS:
        raise RuntimeError(
            f"Backend startup failed: Unrecognized APP_ENV value '{raw_env}'.\n"
            f"Allowed values are strictly: {', '.join(sorted(VALID_APP_ENVS))}.\n"
            "Action required: Set APP_ENV=development for local development or APP_ENV=production for deployment."
        )
    if env == "production":
        token = os.getenv("DEMO_ACCESS_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "Backend startup failed: DEMO_ACCESS_TOKEN must be set when APP_ENV=production.\n"
                "Production authentication must fail closed. Backend refuses to start without an access token."
            )
    return env


def validate_production_configuration() -> None:
    """Backwards-compatible wrapper calling validate_backend_configuration()."""
    validate_backend_configuration()


# Concurrency semaphore
_query_semaphore: Optional[asyncio.Semaphore] = None


def get_query_semaphore() -> asyncio.Semaphore:
    """Return or initialize the asyncio Semaphore bounding concurrent query executions."""
    global _query_semaphore
    if _query_semaphore is None:
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    return _query_semaphore


# In-memory sliding window rate limiter
# NOTE ON RATE LIMIT SCOPE:
# This rate limiter is strictly in-memory and per-process. It is not shared across
# multiple worker processes or container instances, and all limits reset to zero on process restart.
_ip_request_timestamps: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


def determine_client_ip(raw_request: Request) -> str:
    """Determine client IP address, avoiding trust of arbitrary forwarded headers.
    
    Security Policy:
    - Default (TRUST_PROXY_HEADERS=false): Uses direct TCP socket peer IP (raw_request.client.host)
      and ignores X-Forwarded-For headers to prevent spoofed header bypass.
    - Explicit Proxy Trust (TRUST_PROXY_HEADERS=true): Used when deployed behind a known, trusted
      reverse proxy (e.g. Next.js BFF proxy, cloud gateway) that overwrites incoming client headers.
    """
    peer_ip = raw_request.client.host if raw_request.client else "127.0.0.1"
    if TRUST_PROXY_HEADERS and "x-forwarded-for" in raw_request.headers:
        forwarded = raw_request.headers["x-forwarded-for"].split(",")[0].strip()
        if forwarded:
            return forwarded
    return peer_ip


async def check_rate_limit(client_ip: str) -> None:
    """Enforce sliding-window rate limit for demonstration queries.

    Raises:
        HTTPException(429): When client IP exceeds RATE_LIMIT_PER_MINUTE within 60s.
    """
    now = time.time()
    window_start = now - 60.0
    async with _rate_limit_lock:
        timestamps = _ip_request_timestamps[client_ip]
        # Prune timestamps older than 60 seconds
        valid_timestamps = [ts for ts in timestamps if ts > window_start]
        if len(valid_timestamps) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} queries per minute.",
                headers={"Retry-After": "60"},
            )
        valid_timestamps.append(now)
        _ip_request_timestamps[client_ip] = valid_timestamps


def verify_demonstration_access(authorization: Optional[str] = Header(None)) -> None:
    """Verify Bearer token if DEMO_ACCESS_TOKEN is configured in environment.

    Operational Behavior:
    - If DEMO_ACCESS_TOKEN is empty or unset: Allows open local development access.
    - If DEMO_ACCESS_TOKEN is set: Rejects requests without valid 'Bearer <token>'.
    """
    expected_token = os.getenv("DEMO_ACCESS_TOKEN", "").strip()
    if not expected_token:
        return  # Open development mode

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Bearer token required for demonstration access.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid demonstration access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler executing startup validation."""
    validate_production_configuration()
    yield


app = FastAPI(
    title="Personal AI Knowledge Assistant API",
    description="REST backend exposing evidence-grounded RAG query pipeline for Next.js frontend.",
    version="1.0.0",
    lifespan=lifespan,
)

# Request size limiter middleware (16 KB bounded limit)
@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    # Check Content-Length header if present
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return Response(
                    content=json.dumps({
                        "detail": f"Request body too large. Maximum allowed size is {MAX_REQUEST_BODY_BYTES} bytes."
                    }),
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    media_type="application/json",
                )
        except ValueError:
            pass

    # For POST requests, inspect actual streamed body length
    if request.method == "POST":
        body = await request.body()
        if len(body) > MAX_REQUEST_BODY_BYTES:
            return Response(
                content=json.dumps({
                    "detail": f"Request body too large. Maximum allowed size is {MAX_REQUEST_BODY_BYTES} bytes."
                }),
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                media_type="application/json",
            )

    return await call_next(request)


# Configure CORS strictly for the configured frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field(..., description="Service liveness indicator ('ok')")
    service: str = Field(..., description="Service name identifier")


class ReadyResponse(BaseModel):
    """Readiness probe response."""
    status: str = Field(..., description="Service readiness indicator ('ready' or 'unready')")
    service: str = Field(..., description="Service identifier")
    document_chunks: int = Field(..., description="Number of active documents in the Gemini File Search store")
    model: str = Field(..., description="Configured Gemini generation model")
    gemini_key_configured: bool = Field(
        ...,
        description=(
            "Whether GEMINI_API_KEY is configured in the environment. Note: confirms key presence only; "
            "does not guarantee valid credentials, quota availability, or a successful model request."
        ),
    )


class QueryRequest(BaseModel):
    """Payload for RAG question answering."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Non-empty user query string up to 500 characters to search and answer from evidence.",
        examples=["What embedding model and vector database were used in Project Nexus?"],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=10,
        description="Number of candidate evidence chunks to retrieve (1 to 10).",
    )

    @field_validator("question")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question cannot be empty or whitespace only.")
        return stripped


class CitationItem(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int] = None


class EvidenceItem(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    distance: Optional[float] = None  # Cosine distance (smaller = closer)


class TimingsItem(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    """Structured response conforming to the existing RAG pipeline result contract."""
    question: str
    answer: str
    status: str = Field(..., description="answered | abstain | partial | conflict | error")
    missing_information: Optional[str] = None
    citations: list[CitationItem] = []
    invalid_citations: list[str] = []
    citation_disclaimer: str
    retrieved_evidence: list[EvidenceItem] = []
    timings: TimingsItem
    model: str
    error: Optional[str] = None
    error_details: Optional[dict] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check service liveness",
)
async def health_check() -> HealthResponse:
    """Return basic liveness status.

    NOTE ON LIVENESS VS. READINESS:
    This endpoint verifies that the FastAPI HTTP process is alive and accepting
    connections. It does NOT assert that the persistent Chroma vector index is
    populated, that the embedding model is loaded into memory, or that the OpenAI
    API key is verified and funded. Use /ready to check index and model readiness.
    """
    return HealthResponse(
        status="ok",
        service="personal-ai-knowledge-assistant",
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
    summary="Check service readiness",
)
async def readiness_check() -> ReadyResponse:
    """Verify that the backend is initialized with an index and configured credentials.

    Checks:
    1. GEMINI_API_KEY is non-empty and not the default placeholder.
    2. The Gemini File Search store exists and has active documents.

    NOTE ON CREDENTIAL VALIDITY:
    A configured API key confirms presence in the environment only; it is NOT proof of
    valid credentials, quota availability, or a successful model request.

    Returns HTTP 200 if ready; HTTP 503 if unready.
    """
    if not has_gemini_api_key():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unready: GEMINI_API_KEY is not configured.",
        )

    try:
        ready, message, count = await run_in_threadpool(get_store_status)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unready: Unable to access File Search store ({sanitize_error_message(str(e))}).",
        )
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unready: {message}",
        )

    return ReadyResponse(
        status="ready",
        service="personal-ai-knowledge-assistant",
        document_chunks=count,
        model=GEMINI_MODEL,
        gemini_key_configured=True,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG Pipeline"],
    summary="Query the RAG knowledge assistant",
)
async def query_endpoint(
    request: QueryRequest,
    raw_request: Request,
    _auth: None = Depends(verify_demonstration_access),
) -> QueryResponse:
    """Execute the full RAG pipeline on a question with evidence grounding.

    Deployment Protections:
    1. Verifies Bearer authentication if DEMO_ACCESS_TOKEN is configured.
    2. Enforces sliding window rate limit per client IP.
    3. Acquires concurrency semaphore bounding simultaneous CPU/OpenAI executions.
    4. Dispatches the synchronous pipeline execution to a worker threadpool.
    5. Reuses the in-process singleton SentenceTransformer embedding model.
    6. Retrieves top_k chunks from the persistent Chroma collection.
    7. Generates an evidence-grounded answer citing strictly verified chunk IDs.
    8. Returns structured results preserving answered, abstain, partial, conflict,
       and error distinctions.
    """
    # Rate limit check per client IP (uses direct socket IP unless TRUST_PROXY_HEADERS=true)
    client_ip = determine_client_ip(raw_request)
    await check_rate_limit(client_ip)

    # Concurrency limit check
    sem = get_query_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is handling maximum concurrent queries. Please retry in a few moments.",
            headers={"Retry-After": "5"},
        )

    try:
        # Offload synchronous retrieval and OpenAI call to threadpool
        result = await run_in_threadpool(
            query_pipeline,
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as e:
        # Catch unexpected pipeline-level unhandled exceptions and sanitize
        safe_msg = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while executing the query pipeline.",
        ) from None
    finally:
        sem.release()

    return QueryResponse(**result)


def main() -> None:
    """Launch the application server with configurable host and port."""
    validate_backend_configuration()
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "8000").strip() or "8000")

    print("=" * 70)
    print("STARTING PERSONAL AI KNOWLEDGE ASSISTANT API (FASTAPI)")
    print(f"Binding: http://{host}:{port}")
    print(f"Allowed CORS Origins: {ALLOWED_ORIGINS}")
    print(f"Interactive OpenAPI docs available at: http://{host}:{port}/docs")
    print("=" * 70)

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
