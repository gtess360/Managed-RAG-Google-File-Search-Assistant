"use client";

import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { QueryForm } from "../components/QueryForm";
import { ResponseView } from "../components/ResponseView";
import { EvidencePanel } from "../components/EvidencePanel";
import { QueryResponse } from "../types/api";
import { queryAssistant, ApiError } from "../lib/api";

export default function Home() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<{ question: string; top_k: number } | null>(null);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);

  const handleQuery = async (question: string, top_k: number) => {
    setIsLoading(true);
    setErrorMessage(null);
    setLastQuery({ question, top_k });
    setHighlightedChunkId(null);

    try {
      const data = await queryAssistant(question, top_k);
      setResponse(data);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("An unexpected error occurred while communicating with the assistant.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    // Retries occur strictly through explicit user interaction with the Retry button
    if (lastQuery) {
      handleQuery(lastQuery.question, lastQuery.top_k);
    }
  };

  const handleCitationClick = (chunkId: string) => {
    setHighlightedChunkId(chunkId);
  };

  const citedChunkIds = new Set(response?.citations.map((c) => c.chunk_id) || []);

  return (
    <>
      <Navbar />

      <main className="main-content">
        <div className="container">
          <section className="hero">
            <h1 className="hero-title" data-testid="hero-title">
              Evidence-Grounded Knowledge Assistant
            </h1>
            <p className="hero-subtitle">
              Ask technical questions about projects and documents. Answers cite retrieved
              vector chunks with explicit refusal on missing evidence.
            </p>
          </section>

          <QueryForm onSubmit={handleQuery} isLoading={isLoading} />

          {/* Network or Validation Error Banner (Only retried on explicit user click) */}
          {errorMessage && (
            <div className="error-card" data-testid="error-banner">
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ flexShrink: 0, marginTop: "2px" }}
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <div style={{ flex: 1 }}>
                <div className="error-title">Query Execution Failed</div>
                <div className="error-message" data-testid="error-message">
                  {errorMessage}
                </div>
              </div>
              <button
                type="button"
                onClick={handleRetry}
                data-testid="retry-button"
                style={{
                  background: "rgba(239, 68, 68, 0.2)",
                  border: "1px solid rgba(239, 68, 68, 0.4)",
                  color: "#fecaca",
                  borderRadius: "6px",
                  padding: "6px 12px",
                  fontSize: "0.82rem",
                  cursor: "pointer",
                  fontWeight: 600,
                  alignSelf: "center",
                }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Results Grid: Left = Answer & Metadata, Right = Evidence Panel */}
          {response && (
            <section className="results-grid" data-testid="results-grid">
              <ResponseView
                response={response}
                onCitationClick={handleCitationClick}
              />

              <EvidencePanel
                evidence={response.retrieved_evidence}
                highlightedChunkId={highlightedChunkId}
                citedChunkIds={citedChunkIds}
              />
            </section>
          )}

          {!response && !isLoading && !errorMessage && (
            <div
              style={{
                textAlign: "center",
                padding: "48px 24px",
                color: "var(--text-subtle)",
                background: "rgba(15, 23, 42, 0.4)",
                borderRadius: "16px",
                border: "1px dashed rgba(255, 255, 255, 0.08)",
              }}
              data-testid="empty-placeholder"
            >
              <div style={{ fontSize: "2rem", marginBottom: "12px" }}>🔍</div>
              <h3 style={{ fontSize: "1.1rem", color: "var(--text-muted)", marginBottom: "6px" }}>
                Ready to Explore Ingested Knowledge
              </h3>
              <p style={{ fontSize: "0.9rem", maxWidth: "520px", margin: "0 auto" }}>
                Select a suggested question above or type your own question to retrieve candidate
                chunks using Google Gemini File Search.
              </p>
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <p>
            Personal AI Knowledge Assistant &bull; Next.js 16 + FastAPI + Google Gemini File Search
          </p>
        </div>
      </footer>
    </>
  );
}
