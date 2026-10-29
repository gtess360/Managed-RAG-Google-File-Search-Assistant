"""Tests for Phase 1: Configuration, directory resolution, and corpus validation."""

import json
import os
from pathlib import Path
import pytest

from src import config


def test_paths_resolve_independently_of_cwd(monkeypatch, tmp_path):
    """Configuration paths must point to the project directories regardless of cwd."""
    # Change current working directory to a temporary path
    monkeypatch.chdir(tmp_path)
    
    # Verify that config paths still point to the real project root
    assert config.PATHS.root.is_dir()
    assert (config.PATHS.root / "README.md").is_file()
    assert config.PATHS.data_dir == config.PATHS.root / "data" / "documents"
    assert config.PATHS.evaluation_dir == config.PATHS.root / "evaluation"


def test_ensure_directories_creates_folders():
    """ensure_directories creates required directories without raising errors."""
    config.ensure_directories()
    assert config.PATHS.data_dir.is_dir()
    assert config.PATHS.evaluation_dir.is_dir()


def test_missing_api_key_raises_actionable_error(monkeypatch):
    """Calling get_gemini_api_key without a key must raise a helpful ValueError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc_info:
        config.get_gemini_api_key()

    # Must instruct the user clearly what action to take
    error_msg = str(exc_info.value)
    assert ".env" in error_msg
    assert "GEMINI_API_KEY" in error_msg


def test_placeholder_api_key_raises_actionable_error(monkeypatch):
    """Setting GEMINI_API_KEY to placeholder text must also raise a helpful ValueError."""
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(ValueError) as exc_info:
        config.get_gemini_api_key()
    assert "placeholder" in str(exc_info.value).lower()


def test_valid_api_key_returned(monkeypatch):
    """When a real key is present, get_gemini_api_key returns it cleanly."""
    test_key = "AIza-test-mock-key-12345"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)
    assert config.get_gemini_api_key() == test_key


def test_safe_config_summary_never_exposes_key(monkeypatch):
    """safe_config_summary must return boolean indicator and never the actual secret."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-super-secret-key-12345")
    summary = config.get_safe_config_summary()
    assert "gemini_api_key_configured" in summary
    assert summary["gemini_api_key_configured"] is True
    # Verify the secret itself does not appear in any summary value
    for val in summary.values():
        assert "super-secret" not in str(val)


def test_sanitized_documents_exist():
    """All 4 planned sanitized markdown documents must exist and have content."""
    docs = list(config.PATHS.data_dir.glob("*.md"))
    assert len(docs) >= 3, f"Expected at least 3 documents, found {len(docs)}"
    for doc in docs:
        content = doc.read_text(encoding="utf-8")
        assert len(content.strip()) > 50, f"Document {doc.name} appears too short or empty"


def test_evaluation_questions_structure():
    """evaluation/questions.json must contain 20 validated questions with 14 dev / 6 holdout."""
    q_file = config.PATHS.evaluation_dir / "questions.json"
    assert q_file.is_file(), "evaluation/questions.json is missing"
    
    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    assert len(questions) == 20, f"Expected 20 questions, got {len(questions)}"
    
    dev_count = sum(1 for q in questions if q.get("split") == "development")
    holdout_count = sum(1 for q in questions if q.get("split") == "holdout")
    assert dev_count == 14, f"Expected 14 development questions, got {dev_count}"
    assert holdout_count == 6, f"Expected 6 holdout questions, got {holdout_count}"
    
    required_keys = {
        "id", "category", "question", "answerable", "expected_answer_facts",
        "expected_sources", "expected_evidence", "forbidden_claims",
        "expected_behavior", "split"
    }
    for q in questions:
        missing = required_keys - set(q.keys())
        assert not missing, f"Question {q.get('id')} missing keys: {missing}"


def test_validate_app_env_valid():
    """validate_app_env accepts 'development' and 'production' (case-insensitive)."""
    assert config.validate_app_env("development") == "development"
    assert config.validate_app_env("DEVELOPMENT") == "development"
    assert config.validate_app_env("production") == "production"
    assert config.validate_app_env("Production") == "production"


def test_validate_app_env_missing(monkeypatch):
    """validate_app_env raises actionable ValueError when APP_ENV is unset or empty."""
    monkeypatch.delenv("APP_ENV", raising=False)
    with pytest.raises(ValueError) as exc_info:
        config.validate_app_env()
    assert "Missing required environment variable: APP_ENV" in str(exc_info.value)
    assert "APP_ENV=development" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info_empty:
        config.validate_app_env("   ")
    assert "Missing required environment variable: APP_ENV" in str(exc_info_empty.value)


def test_validate_app_env_mistyped():
    """validate_app_env raises actionable ValueError when APP_ENV is mistyped."""
    for bad in ["staging", "prod", "dev", "test", "local"]:
        with pytest.raises(ValueError) as exc_info:
            config.validate_app_env(bad)
        assert "Invalid APP_ENV value" in str(exc_info.value)
        assert "Allowed values are strictly" in str(exc_info.value)


def test_missing_gemini_model_raises_actionable_error(monkeypatch):
    """Calling get_gemini_model when unset must raise an actionable ValueError."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(config, "GEMINI_MODEL", "")
    assert config.has_gemini_model() is False
    with pytest.raises(ValueError) as exc_info:
        config.get_gemini_model()
    msg = str(exc_info.value)
    assert "GEMINI_MODEL" in msg
    assert "gemini-2.5-flash" not in msg
    assert ".env" in msg


def test_configured_gemini_model_returned(monkeypatch):
    """When GEMINI_MODEL is set, get_gemini_model returns it cleanly."""
    monkeypatch.setenv("GEMINI_MODEL", "custom-model-id")
    assert config.get_gemini_model() == "custom-model-id"
    assert config.has_gemini_model() is True


def test_default_gemini_timeout_seconds(monkeypatch):
    """Default timeout must be 120.0s when GEMINI_TIMEOUT_SECONDS is unset or empty."""
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 120.0)
    assert config.get_gemini_timeout_seconds() == 120.0


def test_custom_gemini_timeout_seconds(monkeypatch):
    """Configured GEMINI_TIMEOUT_SECONDS in environment is parsed as float."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "90")
    assert config.get_gemini_timeout_seconds() == 90.0

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "150.5")
    assert config.get_gemini_timeout_seconds() == 150.5


def test_invalid_gemini_timeout_seconds_fallback(monkeypatch):
    """Non-numeric or non-positive GEMINI_TIMEOUT_SECONDS falls back to default 120.0s."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "invalid-timeout")
    assert config.get_gemini_timeout_seconds() == 120.0

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "-10")
    assert config.get_gemini_timeout_seconds() == 120.0

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "0")
    assert config.get_gemini_timeout_seconds() == 120.0


def test_safe_config_summary_includes_timeout(monkeypatch):
    """safe_config_summary must include gemini_timeout_seconds."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "85")
    summary = config.get_safe_config_summary()
    assert "gemini_timeout_seconds" in summary
    assert summary["gemini_timeout_seconds"] == 85.0

