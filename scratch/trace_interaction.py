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

print("=== STARTING INTERACTION TRACE ===")
with patch("src.pipeline.query_pipeline", return_value=mock_result) as mock_pipe:
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    
    # 1. Fresh session
    print("1. Fresh session:")
    print("   recent_questions in state:", at.session_state["recent_questions"])
    print("   sidebar captions:", [c.value for c in at.sidebar.caption])
    print("   sidebar buttons:", [b.label for b in at.sidebar.button])
    
    # 2. Type question and click Ask assistant
    at.text_area[0].input("What database does Project Nexus use?")
    ask_btn = [b for b in at.button if "Ask assistant" in b.label or "Send" in b.label or "submit" in b.key.lower()][0]
    print(f"2. Clicking '{ask_btn.label}'...")
    at = ask_btn.click().run()
    
    print("3. After submission:")
    print("   Pipeline call count:", mock_pipe.call_count)
    print("   recent_questions in state:", at.session_state["recent_questions"])
    print("   sidebar captions:", [c.value for c in at.sidebar.caption])
    print("   sidebar buttons:", [b.label for b in at.sidebar.button])
    print("   Answer visible:", any("ChromaDB" in m.value for m in at.markdown))
    
    # 4. Subsequent rerun (e.g. user toggles something or plain rerun)
    at = at.run()
    print("4. After subsequent rerun:")
    print("   recent_questions in state:", at.session_state["recent_questions"])
    print("   sidebar captions:", [c.value for c in at.sidebar.caption])
    print("   sidebar buttons:", [b.label for b in at.sidebar.button])
