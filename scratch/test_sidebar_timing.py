import sys
sys.path.insert(0, ".")
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

mock_result = {
    "question": "What database does Project Nexus use?",
    "answer": "Project Nexus uses ChromaDB.",
    "status": "answered",
    "missing_information": None,
    "citations": [{"chunk_id": "c1", "source": "nexus.md", "page": 1}],
    "invalid_citations": [],
    "retrieved_evidence": [{"chunk_id": "c1", "text": "evidence", "source": "nexus.md", "page": 1}],
    "timings": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
    "model": "gemini-3.5-flash-lite",
    "error": None,
}

print("=== RUNNING TIMING TEST ===")
with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    
    # Check initial sidebar
    print("Initial sidebar buttons:", [b.label for b in at.sidebar.button])
    print("Initial sidebar captions:", [c.value for c in at.sidebar.caption])
    
    # Enter question and submit
    at.text_area[0].input("What database does Project Nexus use?")
    ask_btn = [b for b in at.button if "Ask assistant" in b.label or "Send" in b.label or "submit" in b.key.lower()][0]
    ask_btn.click().run()
    
    print("After submission pipeline call count:", mock_pipe.call_count)
    print("After submission recent_questions:", at.session_state["recent_questions"])
    print("After submission sidebar buttons:", [b.label for b in at.sidebar.button])
    print("After submission sidebar captions:", [c.value for c in at.sidebar.caption])

