// frontend/src/App.jsx
import React, { useState } from "react";
import ChartView from "./ChartView";

const API_BASE = import.meta.env.VITE_API_URL || "http://172.18.181.41:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [chartOverride, setChartOverride] = useState(null);

  async function submitQuestion(e) {
    e?.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || res.status);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    }
    setLoading(false);
  }

  return (
    <div style={{ padding: 24, fontFamily: "system-ui, Arial" }}>
      <h1>Hostel BI — NL → SQL Agent</h1>

      <form onSubmit={submitQuestion}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          style={{ width: "100%", fontSize: 16 }}
          placeholder="Ask e.g., 'Show total seats per gender and block'"
        />
        <div style={{ marginTop: 8 }}>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Thinking..." : "Ask"}
          </button>
          <button type="button" onClick={() => { setQuestion(""); setResult(null); setError(null); }}>
            Clear
          </button>
        </div>
      </form>

      <hr />

      {error && (
        <div style={{ color: "crimson" }}>
          <strong>Error:</strong> {String(error)}
        </div>
      )}

      {result && (
        <div>
          <h2>Execution Steps</h2>

          <section>
            <h3>1) Prompt sent to LLM</h3>
            <pre style={{ whiteSpace: "pre-wrap", background: "#f5f5f5", padding: 12 }}>
              {result.prompt}
            </pre>
          </section>

          <section>
            <h3>2) Raw LLM response</h3>
            <pre style={{ whiteSpace: "pre-wrap", background: "#fff8dc", padding: 12 }}>
              {result.llm_raw}
            </pre>
          </section>

          <section>
            <h3>3) Extracted SQL</h3>
            <pre style={{ whiteSpace: "pre-wrap", background: "#eef", padding: 12 }}>
              {result.sql || "(no SQL)"}
            </pre>
            <div>
              <strong>Safety check:</strong>{" "}
              {result.safety_ok ? "PASSED" : "FAILED"}
            </div>
          </section>

          <section>
            <h3>4) Chart (auto-detected)</h3>
            <div style={{ marginBottom: 8 }}>
              <label>Force chart: </label>
              <select value={chartOverride || ""} onChange={(e) => setChartOverride(e.target.value || null)}>
                <option value="">Auto</option>
                <option value="bar">Bar</option>
                <option value="pie">Pie</option>
                <option value="line">Line</option>
              </select>
            </div>
            <ChartView dataRows={result.data} chartTypeOverride={chartOverride} />
          </section>

          <section>
            <h3>5) Query result</h3>
            {result.error ? (
              <div style={{ color: "crimson" }}>
                <strong>Execution error:</strong> {result.error}
              </div>
            ) : result.data && result.data.length ? (
              <table border="1" cellPadding="6" style={{ borderCollapse: "collapse", marginTop: 8 }}>
                <thead>
                  <tr>
                    {Object.keys(result.data[0]).map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.data.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((v, j) => (
                        <td key={j}>{String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div>No rows returned.</div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

