"use client";

import React, { useEffect, useRef } from "react";
import { EvidenceItem } from "../types/api";

interface EvidencePanelProps {
  evidence: EvidenceItem[];
  highlightedChunkId: string | null;
  citedChunkIds: Set<string>;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  evidence,
  highlightedChunkId,
  citedChunkIds,
}) => {
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    if (highlightedChunkId && cardRefs.current[highlightedChunkId]) {
      cardRefs.current[highlightedChunkId]?.scrollIntoView?.({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [highlightedChunkId]);

  if (!evidence || evidence.length === 0) {
    return (
      <div className="evidence-panel">
        <div className="evidence-header">
          <div className="evidence-title">
            <span>Retrieved Evidence</span>
          </div>
          <span className="evidence-count-badge">0 chunks</span>
        </div>
        <p style={{ color: "var(--text-subtle)", fontSize: "0.9rem" }}>
          No evidence chunks retrieved for this query.
        </p>
      </div>
    );
  }

  return (
    <div className="evidence-panel" data-testid="evidence-panel">
      <div className="evidence-header">
        <div className="evidence-title">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ color: "#818cf8" }}
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <span>Retrieved Evidence</span>
        </div>
        <span className="evidence-count-badge">{evidence.length} chunks retrieved</span>
      </div>

      <div
        style={{
          fontSize: "0.78rem",
          color: "var(--text-subtle)",
          background: "rgba(15, 23, 42, 0.4)",
          border: "1px solid var(--bg-card-border)",
          borderRadius: "6px",
          padding: "8px 12px",
          lineHeight: "1.4",
        }}
        data-testid="distance-disclaimer"
      >
        Lower distance means a closer vector match; it does not establish answer correctness.
      </div>

      <div className="evidence-list">
        {evidence.map((item) => {
          const isCited = citedChunkIds.has(item.chunk_id);
          const isHighlighted = highlightedChunkId === item.chunk_id;

          return (
            <div
              key={item.chunk_id}
              ref={(el) => {
                cardRefs.current[item.chunk_id] = el;
              }}
              className={`evidence-card ${isHighlighted ? "highlighted" : ""}`}
              id={`evidence-${item.chunk_id}`}
              data-testid={`evidence-card-${item.chunk_id}`}
            >
              <div className="evidence-card-top">
                <div className="chunk-id-tag">
                  <span>#{item.chunk_id}</span>
                  {isCited && (
                    <span className="cited-indicator" data-testid={`cited-${item.chunk_id}`}>
                      Cited in answer text
                    </span>
                  )}
                </div>

                <div
                  className="distance-badge"
                  title="Relevance distance metric"
                  data-testid={`dist-${item.chunk_id}`}
                >
                  <span>Cosine Distance: </span>
                  <strong style={{ color: "#f8fafc" }}>
                    {item.distance !== null && item.distance !== undefined
                      ? item.distance.toFixed(4)
                      : "N/A"}
                  </strong>
                </div>
              </div>

              <div className="evidence-source-row">
                <span className="source-tag" title="Source document">
                  📁 {item.source}
                </span>
                {item.page !== null && item.page !== undefined && (
                  <span className="source-tag">📄 Page {item.page}</span>
                )}
                {item.chunk_index !== null && item.chunk_index !== undefined && (
                  <span className="source-tag">Chunk #{item.chunk_index}</span>
                )}
              </div>

              <div className="evidence-text" data-testid={`evidence-text-${item.chunk_id}`}>
                {item.text}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
