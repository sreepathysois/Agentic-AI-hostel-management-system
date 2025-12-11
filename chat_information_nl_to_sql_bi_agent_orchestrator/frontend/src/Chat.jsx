// frontend/src/Chat.jsx
import React, { useState, useRef, useEffect } from "react";
import ChartView from "./ChartView";

const API_BASE = import.meta.env.VITE_API_URL || "http://172.18.181.41:8000";

function DebugPanel({ meta }) {
  if (!meta) return null;

  const ragUsed = !!meta.kb_hits || !!meta.sources || meta.type === "informational" && meta.kb_hits?.length;
  const hits = meta.kb_hits || [];
  const ragPrompt = meta.rag_prompt || meta.ragPrompt || meta.ragPromptText;
  const llmPrompt = meta.llm_prompt || meta.llmPrompt || meta.llmPromptText;

  return (
    <details style={{ marginTop: 8 }}>
      <summary style={{ cursor: "pointer" }}>Debug (show)</summary>
      <div style={{ padding: 8, background: "#f6f8fa", marginTop: 8 }}>
        <div style={{ fontSize: 13, marginBottom: 6 }}>
          <strong>RAG used:</strong> {ragUsed ? "Yes" : "No"}
        </div>

        {hits && hits.length > 0 && (
          <>
            <div style={{ fontSize: 13, marginBottom: 6 }}>
              <strong>Top-K hits:</strong> {hits.length}
            </div>
            <div style={{ maxHeight: 220, overflow: "auto", border: "1px solid #e1e4e8", padding: 8, borderRadius: 6 }}>
              {hits.map((h, i) => (
                <div key={i} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, color: "#333" }}>
                    <strong>[{i+1}]</strong> <span style={{ color: "#666" }}>{h.source || h.source}</span> {h.score != null && <em style={{ marginLeft: 8 }}>score: {Number(h.score).toFixed(3)}</em>}
                  </div>
                  <pre style={{ whiteSpace: "pre-wrap", fontSize: 13, marginTop: 6 }}>{h.text}</pre>
                </div>
              ))}
            </div>
          </>
        )}

        {(!hits || hits.length === 0) && (
          <div style={{ fontSize: 13, color: "#666" }}>No RAG hits returned by backend.</div>
        )}

        {ragPrompt && (
          <>
            <div style={{ marginTop: 8, fontSize: 13 }}><strong>Assembled RAG prompt (context passed to LLM):</strong></div>
            <pre style={{ whiteSpace: "pre-wrap", maxHeight: 240, overflow: "auto" }}>{ragPrompt}</pre>
          </>
        )}

        {llmPrompt && (
          <>
            <div style={{ marginTop: 8, fontSize: 13 }}><strong>LLM prompt (final prompt sent to LLM):</strong></div>
            <pre style={{ whiteSpace: "pre-wrap", maxHeight: 240, overflow: "auto" }}>{llmPrompt}</pre>
          </>
        )}

        <div style={{ marginTop: 8, fontSize: 12 }}>
          <details>
            <summary>Raw debug JSON</summary>
            <pre style={{ whiteSpace: "pre-wrap", maxHeight: 300, overflow: "auto" }}>
              {JSON.stringify(meta, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    </details>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([]); // {role, text, meta?}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const boxRef = useRef(null);

  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [messages, loading]);

  async function send() {
    if (!input.trim()) return;
    const text = input.trim();
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      // include debug flag in body
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, debug: true }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || res.status);
      }
      const data = await res.json();
      // add assistant message with meta
      setMessages((m) => [...m, { role: "assistant", text: data.answer || data.message || "", meta: data }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: "Error: " + String(err) }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 920, margin: "0 auto", padding: 12 }}>
      <div ref={boxRef} style={{ height: 520, overflow: "auto", border: "1px solid #ddd", padding: 12, borderRadius: 8, background: "#fff" }}>
        {messages.length === 0 && <div style={{ color: "#666" }}>Type a question to start (e.g., "How many seats are available in block 221?")</div>}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 18 }}>
            <div style={{ fontWeight: "bold", textTransform: "capitalize" }}>{m.role}</div>
            <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{m.text}</div>

            {m.meta && m.meta.type === "bi" && (
              <div style={{ marginTop: 8, background: "#fafafa", padding: 10, borderRadius: 6 }}>
                <h4 style={{ margin: "4px 0" }}>SQL</h4>
                <pre style={{ whiteSpace: "pre-wrap" }}>{m.meta.sql}</pre>

                <h4 style={{ margin: "8px 0 4px" }}>Results (chart)</h4>
                <ChartView dataRows={m.meta.data} />

                <h4 style={{ margin: "8px 0 4px" }}>Raw rows</h4>
                {m.meta.data && m.meta.data.length ? (
                  <table border="1" cellPadding="6" style={{ borderCollapse: "collapse", marginTop: 8 }}>
                    <thead>
                      <tr>{Object.keys(m.meta.data[0]).map((k) => <th key={k}>{k}</th>)}</tr>
                    </thead>
                    <tbody>
                      {m.meta.data.map((r, idx) => (
                        <tr key={idx}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div>No rows returned.</div>}
              </div>
            )}

            {m.meta && m.meta.type !== "bi" && (
              <>
                {/* Show sources & passages (if present) */}
                {m.meta.sources && (
                  <div style={{ marginTop: 8, fontSize: 13, color: "#444" }}>
                    <strong>Sources:</strong>{" "}
                    {m.meta.sources.map((s, idx) => <span key={idx} style={{ marginRight: 8 }}>[{s.idx || s.id || idx+1}] {s.source}</span>)}
                  </div>
                )}
                {m.meta.kb_hits && (
                  <details style={{ marginTop: 8 }}>
                    <summary>Show supporting passages</summary>
                    <div style={{ marginTop: 8 }}>
                      {m.meta.kb_hits.map((h, j) => (
                        <div key={j} style={{ marginBottom: 10, background: "#fff", padding: 8, borderRadius: 4, border: "1px solid #eee" }}>
                          <div style={{ fontSize: 12, color: "#666" }}><strong>Source:</strong> {h.source} (id:{h.id}) <em style={{ marginLeft: 8 }}>score: {h.score != null ? Number(h.score).toFixed(3) : "n/a"}</em></div>
                          <pre style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{h.text}</pre>
                        </div>
                      ))}
                    </div>
                  </details>
                )}

                {/* Debug panel shows rag info and prompt contents if present */}
                <DebugPanel meta={m.meta} />
              </>
            )}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12 }}>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={3} style={{ width: "100%", padding: 8 }} />
        <div style={{ marginTop: 8 }}>
          <button onClick={send} disabled={loading || !input.trim()} style={{ padding: "8px 14px" }}>{loading ? "Thinking..." : "Send"}</button>
          <button onClick={() => setInput("")} style={{ marginLeft: 8, padding: "8px 12px" }}>Clear</button>
        </div>
      </div>
    </div>
  );
}

