import sys
sys.path.insert(0, ".")
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest
import pytest

mock_result = {
    "question": "What database does Project Nexus use?",
    "answer": "Project Nexus uses ChromaDB.",
    "status": "answered",
    "missing_information": None,
    "citations": [{"chunk_id": "c1", "source": "s1.md", "page": 1}],
    "invalid_citations": [],
    "retrieved_evidence": [{"chunk_id": "c1", "text": "evidence", "source": "s1.md", "page": 1}],
    "timings": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
    "model": "gemini-3.5-flash-lite",
    "error": None,
}

with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    at.text_area[0].input("What database does Project Nexus use?")
    # Submit
    submit_btn = [b for b in at.button if "Ask assistant" in b.label or "Send" in b.label or "submit" in b.key.lower()][0]
    submit_btn.click().run()

    print("Pipeline call count:", mock_pipe.call_count)
    print("last_result in session_state:", at.session_state["last_result"] is not None)
    print("recent_questions:", at.session_state.get("recent_questions"))
    sidebar_buttons = [b.label for b in at.sidebar.button]
    print("sidebar buttons:", sidebar_buttons)
