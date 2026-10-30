import streamlit as st

# Pre-populate session state before loading the app
if "last_result" not in st.session_state or st.session_state["last_result"] is None:
    st.session_state["last_result"] = {
        "question": "What embedding model and vector database were used in Project Nexus?",
        "answer": (
            "In Project Nexus, the documents were embedded using **OpenAI's text-embedding-3-small model**, "
            "and the vector database used was **ChromaDB**.\n\n"
            "This setup enables efficient semantic search and retrieval of relevant document chunks for grounded answers."
        ),
        "status": "answered",
        "missing_information": None,
        "citations": [
            {
                "chunk_id": "nexus_chunk_001",
                "source": "document_assistant_project.md",
                "page": 1,
            },
            {
                "chunk_id": "nexus_chunk_002",
                "source": "churn_project.md",
                "page": 1,
            },
        ],
        "invalid_citations": [],
        "citation_disclaimer": "Citations verified from store metadata.",
        "retrieved_evidence": [
            {
                "chunk_id": "nexus_chunk_001",
                "text": "Planned vector database: ChromaDB ... Embeddings: OpenAI text-embedding-3-small ...",
                "source": "document_assistant_project.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.1245,
            },
            {
                "chunk_id": "nexus_chunk_002",
                "text": "Document ingestion uses OpenAI embeddings with ChromaDB for storage and retrieval ...",
                "source": "churn_project.md",
                "page": 1,
                "chunk_index": 0,
                "distance": 0.1872,
            },
        ],
        "timings": {"retrieval_ms": 14.2, "generation_ms": 280.5, "total_ms": 294.7},
        "model": "gemini-3.5-flash-lite",
        "error": None,
    }
    st.session_state["question_input"] = "What embedding model and vector database were used in Project Nexus?"
    st.session_state["questions_used"] = 2
    st.session_state["recent_questions"] = [
        "What embedding model and vector database were used in Project Nexus?",
        "Explain the infrastructure for Project Aurora.",
    ]
    st.session_state["show_more_examples"] = True

import streamlit_app
streamlit_app.main()
