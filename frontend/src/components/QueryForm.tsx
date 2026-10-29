"use client";

import React, { useState } from "react";

interface QueryFormProps {
  onSubmit: (question: string, top_k: number) => void;
  isLoading: boolean;
  initialQuestion?: string;
}

const SAMPLE_PROMPTS = [
  "What embedding model and vector database were used in Project Nexus?",
  "What was the key achievement of the Quantum Computing Initiative?",
  "Who founded Apple Inc and when was the iPhone announced?", // Abstention probe
  "What are the main findings of the evaluation benchmark report?",
];

export const QueryForm: React.FC<QueryFormProps> = ({
  onSubmit,
  isLoading,
  initialQuestion = "",
}) => {
  const [question, setQuestion] = useState(initialQuestion);
  // Default retrieval candidates is 5, consistent with current recheck default.
  // NOTE: This frontend default is statically and separately configured here
  // rather than dynamically shared from the backend's DEFAULT_TOP_K environment variable.
  const [topK, setTopK] = useState(5);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Guard against duplicate submission while loading
    if (isLoading) return;
    const trimmed = question.trim();
    if (!trimmed) {
      setValidationError("Please enter a question to search and answer.");
      return;
    }
    setValidationError(null);
    onSubmit(trimmed, topK);
  };

  const handleSelectSample = (sample: string) => {
    if (isLoading) return;
    setQuestion(sample);
    setValidationError(null);
    onSubmit(sample, topK);
  };

  return (
    <div className="search-card" data-testid="search-card">
      <form onSubmit={handleSubmit} className="search-form">
        <div className="input-group">
          <svg
            className="search-icon"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            className="search-input"
            data-testid="search-input"
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value);
              if (validationError) setValidationError(null);
            }}
            placeholder="Ask a question grounded in your ingested portfolio and technical documents..."
            disabled={isLoading}
            autoFocus
          />
        </div>

        {validationError && (
          <div
            style={{ color: "#f87171", fontSize: "0.85rem", paddingLeft: "8px" }}
            data-testid="validation-error"
          >
            ⚠ {validationError}
          </div>
        )}

        <div className="controls-bar">
          <div className="top-k-group">
            <span>Retrieval candidates (top_k):</span>
            <div className="top-k-slider-container">
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                className="top-k-slider"
                data-testid="top-k-slider"
                disabled={isLoading}
              />
              <span className="top-k-badge" data-testid="top-k-badge">
                {topK}
              </span>
            </div>
          </div>

          <button
            type="submit"
            className="btn-submit"
            data-testid="submit-btn"
            disabled={isLoading || !question.trim()}
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                <span>Retrieving & Generating...</span>
              </>
            ) : (
              <>
                <span>Ask Assistant</span>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="5" y1="12" x2="19" y2="12" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </>
            )}
          </button>
        </div>

        {isLoading && (
          <div
            style={{
              fontSize: "0.82rem",
              color: "#a5b4fc",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              paddingTop: "6px",
            }}
            data-testid="loading-indicator"
          >
            <span className="status-dot online" style={{ width: "6px", height: "6px" }} />
            <span>
              Searching File Search store and generating grounded answer with Gemini... (typically 2–5s)
            </span>
          </div>
        )}

        <div className="sample-prompts">
          <div className="sample-prompts-label">Suggested Test Questions:</div>
          <div className="sample-prompts-list">
            {SAMPLE_PROMPTS.map((prompt, idx) => (
              <button
                key={idx}
                type="button"
                className="sample-chip"
                onClick={() => handleSelectSample(prompt)}
                disabled={isLoading}
                data-testid={`sample-chip-${idx}`}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
};
