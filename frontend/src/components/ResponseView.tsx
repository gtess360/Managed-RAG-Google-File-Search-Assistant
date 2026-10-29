"use client";

import React from "react";
import { QueryResponse, QueryStatus } from "../types/api";
import { MarkdownAnswer } from "./MarkdownAnswer";

interface ResponseViewProps {
  response: QueryResponse;
  onCitationClick: (chunkId: string) => void;
}

export const ResponseView: React.FC<ResponseViewProps> = ({
  response,
  onCitationClick,
}) => {
  const getStatusDisplay = (status: QueryStatus) => {
    switch (status) {
      case "answered":
        return {
          icon: "✓",
          label: "Answered",
          className: "answered",
        };
      case "abstain":
        return {
          icon: "⚠",
          label: "Abstained (Insufficient Evidence)",
          className: "abstain",
        };
      case "partial":
        return {
          icon: "◐",
          label: "Partial Answer",
          className: "partial",
        };
      case "conflict":
        return {
          icon: "⚡",
          label: "Contradiction / Conflict Detected",
          className: "conflict",
        };
      case "error":
      default:
        return {
          icon: "✕",
          label: "Pipeline Error",
          className: "error",
        };
    }
  };

  const statusMeta = getStatusDisplay(response.status);
  const validCitationIds = new Set(response.citations.map((c) => c.chunk_id));

  return (
    <div className="answer-card" data-testid="answer-card">
      <div className="answer-header">
        <span
          className={`status-pill ${statusMeta.className}`}
          data-testid="status-pill"
        >
          <span>{statusMeta.icon}</span>
          <span>{statusMeta.label}</span>
        </span>

        <div className="meta-chips">
          <span className="meta-chip" title="Total Roundtrip Latency">
            ⏱ {response.timings.total_ms.toFixed(0)} ms total
          </span>
          <span className="meta-chip" title="File Search Retrieval Latency">
            🔍 {response.timings.retrieval_ms.toFixed(0)} ms retrieval
          </span>
          <span className="meta-chip" title="Gemini LLM Generation Latency">
            🤖 {response.timings.generation_ms.toFixed(0)} ms generation
          </span>
          <span className="meta-chip" title="Model Identifier" data-testid="model-badge">
            {response.model}
          </span>
        </div>
      </div>

      {/* Missing Information Callout */}
      {response.missing_information && (
        <div className="missing-info-box" data-testid="missing-info-box">
          <span style={{ fontSize: "1.1rem" }}>ℹ</span>
          <div>
            <strong style={{ display: "block", marginBottom: "2px" }}>
              Missing Information:
            </strong>
            <span>{response.missing_information}</span>
          </div>
        </div>
      )}

      {/* Error detail banner if status is error */}
      {response.status === "error" && response.error && (
        <div
          className="missing-info-box"
          style={{
            background: "rgba(239, 68, 68, 0.1)",
            borderColor: "rgba(239, 68, 68, 0.3)",
            color: "#fca5a5",
          }}
          data-testid="pipeline-error-detail"
        >
          <span>✕</span>
          <div>
            <strong>Error Details:</strong>
            <p style={{ fontSize: "0.88rem", marginTop: "4px" }}>{response.error}</p>
          </div>
        </div>
      )}

      {/* Answer Body with Clickable Citations */}
      <MarkdownAnswer
        content={response.answer}
        onCitationClick={onCitationClick}
        validCitationIds={validCitationIds}
      />

      {/* Unmatched Citations Warning if LLM cited IDs not in retrieved evidence */}
      {response.invalid_citations && response.invalid_citations.length > 0 && (
        <div
          className="missing-info-box"
          style={{
            background: "rgba(239, 68, 68, 0.1)",
            borderColor: "rgba(239, 68, 68, 0.3)",
            color: "#fca5a5",
          }}
          data-testid="invalid-citations-box"
        >
          <span>⚠</span>
          <div>
            <strong>Unmatched Citations Rejected:</strong>
            <p style={{ fontSize: "0.85rem", marginTop: "4px" }}>
              The model cited IDs not present in retrieved evidence:{" "}
              <code>{response.invalid_citations.join(", ")}</code>. These were filtered out.
            </p>
          </div>
        </div>
      )}

      {/* Citations matched to retrieved evidence */}
      {response.citations && response.citations.length > 0 && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "8px",
            paddingTop: "8px",
            borderTop: "1px solid var(--bg-card-border)",
          }}
          data-testid="citations-section"
        >
          <span style={{ fontSize: "0.78rem", color: "var(--text-subtle)", fontWeight: 600 }}>
            CITATIONS MATCHED TO RETRIEVED EVIDENCE:
          </span>
          {response.citations.map((c) => (
            <button
              key={c.chunk_id}
              type="button"
              className="citation-badge"
              onClick={() => onCitationClick(c.chunk_id)}
              data-testid={`citation-btn-${c.chunk_id}`}
            >
              📄 {c.chunk_id} {c.page ? `(p. ${c.page})` : ""}
            </button>
          ))}
        </div>
      )}

      {/* Citation Validity Disclaimer */}
      {response.citation_disclaimer && (
        <div className="disclaimer-box" data-testid="citation-disclaimer">
          <strong>Citation Disclaimer: </strong>
          <span>
            {response.citation_disclaimer} Citation-ID validation does not prove factual support.
          </span>
        </div>
      )}
    </div>
  );
};
