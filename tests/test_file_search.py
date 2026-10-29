"""Offline tests for the Gemini File Search integration (no network)."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

from src import file_search as fs
from src.pipeline import query_pipeline


@pytest.fixture(autouse=True)
def configure_test_model(monkeypatch):
    """Ensure tests run with an explicit model configured by default."""
    monkeypatch.setenv("GEMINI_MODEL", "mock-gemini-model")
    monkeypatch.setattr(fs, "GEMINI_MODEL", "mock-gemini-model")


def _response(text, chunks=(), supports=()):
    gm = NS(
        grounding_chunks=[
            NS(retrieved_context=NS(title=t, text=x, page_number=None)) for t, x in chunks
        ],
        grounding_supports=[NS(grounding_chunk_indices=list(i)) for i in supports],
    )
    return NS(text=text, candidates=[NS(grounding_metadata=gm)])


class _FakeClient:
    def __init__(self, response=None, store_get_fail=False):
        self._response = response
        self.models = NS(generate_content=lambda **kw: self._response)
        
        def _get(name):
            if store_get_fail:
                raise RuntimeError("Store not found (404)")
            return NS(name=name, display_name="test-store", active_documents_count=4)

        self.file_search_stores = NS(
            get=_get,
            list=lambda: [NS(name="fileSearchStores/other-store", display_name=fs.FILE_SEARCH_STORE_DISPLAY_NAME)],
            create=lambda config: NS(name="fileSearchStores/newly-created", display_name=config.get("display_name")),
            documents=NS(
                list=lambda parent: [],
                delete=lambda name, config=None: None,
            ),
        )


def test_parse_structured_text():
    status, answer, missing = fs.parse_structured_text(
        "STATUS: partial\nMISSING: latency numbers\nANSWER:\nNexus uses Chroma."
    )
    assert (status, answer, missing) == ("partial", "Nexus uses Chroma.", "latency numbers")


def test_answered_with_citations():
    resp = _response(
        "STATUS: answered\nMISSING: NONE\nANSWER:\nProject Nexus uses Chroma.",
        chunks=[("nexus.md", "Nexus uses Chroma.")],
        supports=[[0]],
    )
    r = fs.ask("What does Nexus use?", client=_FakeClient(resp), store_name="fileSearchStores/x")
    assert r["status"] == "answered"
    assert r["citations"][0]["source"] == "nexus.md"
    assert r["retrieved_evidence"][0]["distance"] is None


def test_ungrounded_answer_is_error():
    resp = _response("STATUS: answered\nMISSING: NONE\nANSWER:\nMade-up claim.")
    r = fs.ask("q", client=_FakeClient(resp), store_name="fileSearchStores/x")
    assert r["status"] == "error"


def test_abstain():
    resp = _response(f"STATUS: abstain\nMISSING: NONE\nANSWER:\n{fs.INSUFFICIENT_EVIDENCE_PHRASE}")
    r = fs.ask("q", client=_FakeClient(resp), store_name="fileSearchStores/x")
    assert r["status"] == "abstain"
    assert r["answer"] == fs.INSUFFICIENT_EVIDENCE_PHRASE


def test_empty_question_rejected():
    r = query_pipeline("   ")
    assert r["status"] == "error"


def test_ask_missing_model_returns_actionable_error(monkeypatch):
    """When GEMINI_MODEL is empty and no model override is passed, ask returns actionable error."""
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setattr(fs, "GEMINI_MODEL", "")
    r = fs.ask("What is Aurora?", client=_FakeClient(_response("ok")), store_name="fileSearchStores/x")
    assert r["status"] == "error"
    assert "Missing GEMINI_MODEL configuration" in r["error"]


# -----------------------------------------------------------------------------
# Store Selection & Creation Tests (Requirement 1)
# -----------------------------------------------------------------------------

def test_find_store_explicit_env(monkeypatch):
    """Explicit FILE_SEARCH_STORE_NAME in environment is retrieved directly."""
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "fileSearchStores/configured-123")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "fileSearchStores/configured-123")

    client = _FakeClient()
    store = fs.find_store(client)
    assert store is not None
    assert store.name == "fileSearchStores/configured-123"


def test_find_store_cached_file(monkeypatch, tmp_path):
    """Cached store name in file_search_store.json is retrieved when env var is unset."""
    cache_file = tmp_path / "file_search_store.json"
    cache_file.write_text(json.dumps({"store_name": "fileSearchStores/cached-456"}), encoding="utf-8")

    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STATE_FILE", cache_file)

    client = _FakeClient()
    store = fs.find_store(client)
    assert store is not None
    assert store.name == "fileSearchStores/cached-456"


def test_find_store_does_not_select_by_display_name_alone(monkeypatch, tmp_path):
    """Safety: When unconfigured, find_store MUST return None and NOT select a store by display_name."""
    empty_cache = tmp_path / "empty_store.json"
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STATE_FILE", empty_cache)

    # Even though _FakeClient has a store with display_name matching in list(),
    # find_store must not commandeer it.
    client = _FakeClient()
    store = fs.find_store(client)
    assert store is None


def test_find_store_stale_or_missing_returns_none(monkeypatch):
    """When a configured store name cannot be found (404/deleted), find_store returns None."""
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "fileSearchStores/stale-store")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "fileSearchStores/stale-store")

    client = _FakeClient(store_get_fail=True)
    store = fs.find_store(client)
    assert store is None


def test_create_store_explicit_creation_and_caching(monkeypatch, tmp_path):
    """create_store creates a new store via API and caches its name to storage."""
    cache_file = tmp_path / "file_search_store.json"
    monkeypatch.setattr(fs, "FILE_SEARCH_STATE_FILE", cache_file)

    client = _FakeClient()
    store = fs.create_store(client, display_name="project-test-store")
    assert store.name == "fileSearchStores/newly-created"

    assert cache_file.is_file()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["store_name"] == "fileSearchStores/newly-created"


def test_get_or_create_store_raises_when_unconfigured(monkeypatch, tmp_path):
    """get_or_create_store raises actionable ValueError when unconfigured and create_if_missing is False."""
    empty_cache = tmp_path / "empty_store.json"
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STATE_FILE", empty_cache)

    client = _FakeClient()
    with pytest.raises(ValueError) as exc_info:
        fs.get_or_create_store(client, create_if_missing=False)
    assert "--create-store" in str(exc_info.value)


def test_get_or_create_store_creates_when_flag_true(monkeypatch, tmp_path):
    """get_or_create_store explicitly creates store when create_if_missing=True."""
    cache_file = tmp_path / "file_search_store.json"
    monkeypatch.setenv("FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STORE_NAME", "")
    monkeypatch.setattr(fs, "FILE_SEARCH_STATE_FILE", cache_file)

    client = _FakeClient()
    store = fs.get_or_create_store(client, create_if_missing=True)
    assert store.name == "fileSearchStores/newly-created"


# -----------------------------------------------------------------------------
# Document Preservation & Failure Handling Tests (Requirements 2 & 3)
# -----------------------------------------------------------------------------

def test_inspect_documents_categorization(tmp_path):
    """inspect_documents correctly categorizes new, unchanged, changed, and duplicate documents."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    
    f_new = data_dir / "new.md"
    f_new.write_text("new content", encoding="utf-8")
    
    f_unchanged = data_dir / "unchanged.md"
    f_unchanged.write_text("constant", encoding="utf-8")
    
    f_changed = data_dir / "changed.md"
    f_changed.write_text("updated content", encoding="utf-8")

    remote_docs = [
        NS(name="docs/1", display_name="unchanged.md", size_bytes=f_unchanged.stat().st_size),
        NS(name="docs/2", display_name="changed.md", size_bytes=9999),  # size mismatch
        NS(name="docs/3", display_name="orphan.md", size_bytes=50),
    ]

    client = NS(
        file_search_stores=NS(
            documents=NS(list=lambda parent: remote_docs)
        )
    )

    files = [f_new, f_unchanged, f_changed]
    plan = fs.inspect_documents(client, "store123", files, data_dir)

    assert len(plan["new"]) == 1
    assert plan["new"][0][1] == "new.md"

    assert len(plan["unchanged"]) == 1
    assert plan["unchanged"][0][1] == "unchanged.md"

    assert len(plan["changed"]) == 1
    assert plan["changed"][0][1] == "changed.md"

    assert len(plan["remote_only"]) == 1
    assert plan["remote_only"][0].display_name == "orphan.md"


def test_sync_documents_non_destructive_by_default(tmp_path):
    """By default (replace=False), existing remote documents are never deleted."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    doc_file = data_dir / "doc1.md"
    doc_file.write_text("hello world", encoding="utf-8")

    deleted_names = []
    uploaded_files = []

    existing_doc = NS(name="docs/existing-1", display_name="doc1.md", size_bytes=doc_file.stat().st_size)

    client = NS(
        file_search_stores=NS(
            get=lambda name: NS(name=name),
            create=lambda config: NS(name="store-1"),
            upload_to_file_search_store=lambda **kw: uploaded_files.append(kw) or NS(done=True, error=None),
            documents=NS(
                list=lambda parent: [existing_doc],
                delete=lambda name, config=None: deleted_names.append(name),
            ),
        ),
        operations=NS(get=lambda op: op),
    )

    summary = fs.sync_documents(
        data_dir=data_dir,
        replace=False,
        create_store_if_missing=True,
        client=client,
    )

    # In non-destructive mode, the unchanged doc was skipped and NO documents were deleted
    assert len(deleted_names) == 0
    assert summary["removed"] == 0
    assert summary["uploaded"] == 0
    assert summary["skipped"] == 1


def test_sync_documents_upload_failure_preserves_existing(tmp_path):
    """If an upload fails, existing documents are NOT deleted and error is raised."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    doc_new = data_dir / "doc_new.md"
    doc_new.write_text("brand new document", encoding="utf-8")

    existing_doc = NS(name="docs/existing-1", display_name="doc_old.md", size_bytes=100)
    deleted_names = []

    def _fail_upload(**kw):
        raise RuntimeError("Network error during upload")

    client = NS(
        file_search_stores=NS(
            get=lambda name: NS(name=name),
            upload_to_file_search_store=_fail_upload,
            documents=NS(
                list=lambda parent: [existing_doc],
                delete=lambda name, config=None: deleted_names.append(name),
            ),
        ),
        operations=NS(get=lambda op: op),
    )

    # Must raise RuntimeError and NOT treat partial ingestion as complete
    with pytest.raises(RuntimeError) as exc_info:
        fs.sync_documents(
            data_dir=data_dir,
            replace=False,
            create_store_if_missing=True,
            client=client,
            allow_partial=False,
        )

    assert "Ingestion incomplete" in str(exc_info.value)
    # Existing documents were preserved and NOT deleted
    assert len(deleted_names) == 0


def test_sync_documents_indexing_op_error_preserves_existing(tmp_path):
    """If indexing operation completes with op.error, existing documents are preserved and failure raised."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    doc_new = data_dir / "doc_new.md"
    doc_new.write_text("content", encoding="utf-8")

    existing_doc = NS(name="docs/existing-1", display_name="doc_old.md", size_bytes=100)
    deleted_names = []

    client = NS(
        file_search_stores=NS(
            get=lambda name: NS(name=name),
            upload_to_file_search_store=lambda **kw: NS(done=True, error=NS(message="Corrupt file format")),
            documents=NS(
                list=lambda parent: [existing_doc],
                delete=lambda name, config=None: deleted_names.append(name),
            ),
        ),
        operations=NS(get=lambda op: op),
    )

    with pytest.raises(RuntimeError) as exc_info:
        fs.sync_documents(
            data_dir=data_dir,
            replace=False,
            create_store_if_missing=True,
            client=client,
            allow_partial=False,
        )

    assert "Corrupt file format" in str(exc_info.value)
    assert len(deleted_names) == 0


def test_sync_documents_dry_run_reports_plan_without_mutations(tmp_path):
    """dry_run mode reports plan without executing any uploads or deletions."""
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    f1 = data_dir / "f1.md"
    f1.write_text("file 1", encoding="utf-8")

    upload_calls = []
    delete_calls = []

    client = NS(
        file_search_stores=NS(
            get=lambda name: NS(name=name),
            upload_to_file_search_store=lambda **kw: upload_calls.append(kw),
            documents=NS(
                list=lambda parent: [],
                delete=lambda name, config=None: delete_calls.append(name),
            ),
        )
    )

    res = fs.sync_documents(
        data_dir=data_dir,
        dry_run=True,
        create_store_if_missing=True,
        client=client,
    )

    assert res["dry_run"] is True
    assert len(upload_calls) == 0
    assert len(delete_calls) == 0
    assert "new" in res["plan"]
    assert "f1.md" in res["plan"]["new"]


def test_sanitize_error_message_redactions():
    """Verify sanitize_error_message redacts various secret formats."""
    raw = "Failed with AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz and Bearer eyJhbGciOiJIUzI1Ni.test and key=secret123"
    sanitized = fs.sanitize_error_message(raw)
    assert "AIzaSy" not in sanitized
    assert "secret123" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "[REDACTED_TOKEN]" in sanitized
    assert "[REDACTED_KEY]" in sanitized


def test_ask_api_error_preserves_safe_diagnostics():
    """API errors preserve safe diagnostics internally, mask secrets, and suppress answer."""
    from google.genai.errors import ClientError

    def _fail(**kw):
        raise ClientError(
            404,
            {"error": {"message": "models/custom-model with key AIzaSy12345678901234567890 is not found", "status": "NOT_FOUND"}},
        )

    client = NS(
        models=NS(generate_content=_fail),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    r = fs.ask("What is Aurora?", client=client, store_name="fileSearchStores/x")

    assert r["status"] == "error"
    # Insufficient-evidence answer must be SUPPRESSED on error
    assert r["answer"] == ""
    assert r["citation_disclaimer"] == ""

    # Safe diagnostics must be preserved
    assert "error_details" in r
    details = r["error_details"]
    assert details["exception_type"] == "ClientError"
    assert details["code"] == 404
    assert details["model"] == "mock-gemini-model"
    assert "not found" in details["sanitized_reason"]

    # Secret API key must be REDACTED
    assert "AIzaSy12345678901234567890" not in r["error"]
    assert "AIzaSy12345678901234567890" not in details["sanitized_reason"]
    assert "[REDACTED_API_KEY]" in details["sanitized_reason"]


def test_pipeline_validation_error_suppresses_answer():
    """Validation errors in query_pipeline return status='error' with empty answer and details."""
    r = query_pipeline("   ")
    assert r["status"] == "error"
    assert r["answer"] == ""
    assert r["error_details"]["exception_type"] == "ValueError"
    assert r["error_details"]["code"] == 400


# -----------------------------------------------------------------------------
# Timeout Configuration, Propagation & Error Handling Tests
# -----------------------------------------------------------------------------

def test_timeout_default_propagation_in_generate_content(monkeypatch):
    """Default timeout (120s) must be converted to 120,000ms in types.HttpOptions."""
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    captured_configs = []

    resp = _response(
        "STATUS: answered\nMISSING: NONE\nANSWER:\nAnswer text.",
        chunks=[("doc.md", "Chunk text")],
        supports=[[0]],
    )

    client = NS(
        models=NS(generate_content=lambda **kw: captured_configs.append(kw.get("config")) or resp),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    r = fs.ask("Test query?", client=client, store_name="fileSearchStores/test")
    assert r["status"] == "answered"
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config is not None
    assert config.http_options is not None
    assert config.http_options.timeout == 120000


def test_timeout_custom_explicit_argument_to_ms():
    """Explicit timeout parameter (e.g. 45s) must be converted to 45,000ms in HttpOptions."""
    captured_configs = []
    resp = _response(
        "STATUS: answered\nMISSING: NONE\nANSWER:\nAnswer text.",
        chunks=[("doc.md", "Chunk text")],
        supports=[[0]],
    )
    client = NS(
        models=NS(generate_content=lambda **kw: captured_configs.append(kw.get("config")) or resp),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    r = fs.ask("Test query?", client=client, store_name="fileSearchStores/test", timeout=45.0)
    assert r["status"] == "answered"
    assert captured_configs[0].http_options.timeout == 45000


def test_timeout_env_var_propagates_through_pipeline(monkeypatch):
    """GEMINI_TIMEOUT_SECONDS set in environment propagates through query_pipeline to HttpOptions."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "75")
    captured_configs = []
    resp = _response(
        "STATUS: answered\nMISSING: NONE\nANSWER:\nAnswer text.",
        chunks=[("doc.md", "Chunk text")],
        supports=[[0]],
    )
    client = NS(
        models=NS(generate_content=lambda **kw: captured_configs.append(kw.get("config")) or resp),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    with monkeypatch.context() as m:
        m.setattr(fs, "find_store", lambda c: NS(name="fileSearchStores/test"))
        r = query_pipeline("What is Nexus?", client=client)

    assert r["status"] == "answered"
    assert len(captured_configs) == 1
    assert captured_configs[0].http_options.timeout == 75000


def test_get_client_configures_http_options_timeout(monkeypatch):
    """get_client creates client with configured HttpOptions timeout in milliseconds."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSy-fake-key-for-test-12345")
    client = fs.get_client(timeout=90.0)
    assert client._api_client._http_options.timeout == 90000


def test_ask_http_504_gateway_timeout_error_handling():
    """HTTP 504 ServerError returns status='error', clear timeout summary, and empty answer."""
    from google.genai.errors import ServerError

    call_count = 0

    def _fail_504(**kw):
        nonlocal call_count
        call_count += 1
        raise ServerError(
            504,
            {"error": {"code": 504, "message": "Deadline Exceeded", "status": "DEADLINE_EXCEEDED"}},
        )

    client = NS(
        models=NS(generate_content=_fail_504),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    r = fs.ask("What is Aurora?", client=client, store_name="fileSearchStores/x", timeout=120.0)

    # 1. Must NOT retry at application level (bounded to single SDK execution)
    assert call_count == 1

    # 2. Status must be error and answer must be suppressed (NOT insufficient evidence)
    assert r["status"] == "error"
    assert r["answer"] == ""
    assert fs.INSUFFICIENT_EVIDENCE_PHRASE not in r["answer"]
    assert fs.INSUFFICIENT_EVIDENCE_PHRASE not in (r.get("error") or "")
    assert r["citations"] == []
    assert r["retrieved_evidence"] == []

    # 3. Clear timeout summary and diagnostics
    assert "Gemini API request timed out after 120s. Please retry." in r["error"]
    assert "HTTP 504" in r["error"]
    details = r["error_details"]
    assert details["code"] == 504
    assert details["exception_type"] == "ServerError"
    assert details["summary"] == "Gemini API request timed out after 120s. Please retry."


def test_ask_client_timeout_exception_handling():
    """Client-side TimeoutError returns clear timeout summary, status='error', and empty answer."""
    def _fail_timeout(**kw):
        raise TimeoutError("The read operation timed out")

    client = NS(
        models=NS(generate_content=_fail_timeout),
        file_search_stores=NS(get=lambda name: NS(name=name)),
    )

    r = fs.ask("What is Aurora?", client=client, store_name="fileSearchStores/x", timeout=120.0)

    assert r["status"] == "error"
    assert r["answer"] == ""
    assert fs.INSUFFICIENT_EVIDENCE_PHRASE not in r["answer"]
    assert "Gemini API request timed out after 120s. Please retry." in r["error"]
    assert r["error_details"]["summary"] == "Gemini API request timed out after 120s. Please retry."

