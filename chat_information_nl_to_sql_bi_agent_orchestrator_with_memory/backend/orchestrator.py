# orchestrator.py
"""
Updated orchestrator for Agentic HMS (RAG + KB + BI + Session Memory).
Ensures BI responses include a human-friendly assistant answer that also
contains a short textual preview of SQL results so the frontend shows it
directly in the assistant message (not only in the debug panel).

Save as orchestrator.py in your backend (same folder as nl_agent.py, kb_rag.py, memory.py).
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# local modules (assumes running in /app and modules available)
from nl_agent import generate_sql_and_run, SCHEMA_TEXT  # existing NL->SQL agent
from kb_rag import retrieve_top_k  # RAG retrieval helper
from memory import retrieve_memory, add_memory  # session memory helpers

# LLM wrapper (your project uses langchain_openai wrapper)
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# Initialize LLM (tweak model/temperature if desired)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# KB / intent helpers (kept compact; reuse your existing logic)
# -------------------------
KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base")

def load_kb() -> Dict[str, Any]:
    kb = {}
    if not os.path.isdir(KB_PATH):
        return kb
    for fname in ("hostel_types.json", "fees.json", "mess_info.json", "faq.json"):
        path = os.path.join(KB_PATH, fname)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    kb_key = fname.replace(".json", "")
                    kb[kb_key] = json.load(f)
            except Exception:
                kb[kb_key] = None
    return kb

KNOWLEDGE_BASE = load_kb()

BI_KEYWORDS = {
    "show", "list", "count", "total", "per", "by", "how many", "available",
    "vacant", "occupied", "seats", "rooms", "bookings", "pending", "fees",
    "payment", "availability", "seat", "block", "vacancy", "students"
}
PATTERN = re.compile(r"\b(block|seat|room|student|gender|fees|booking|date|month|year|vacant|available)\b", re.I)

def looks_like_data_question(text: str) -> bool:
    t = (text or "").lower()
    if "how many" in t or "count" in t:
        return True
    kw_hits = sum(1 for k in BI_KEYWORDS if k in t)
    pattern_hit = bool(PATTERN.search(text or ""))
    return (kw_hits >= 1 and pattern_hit)

def kb_lookup_struct(question: str) -> Optional[Dict[str, Any]]:
    q = (question or "").lower()
    if not KNOWLEDGE_BASE:
        return None
    if any(word in q for word in ("single", "double", "triple", "attached", "non ac", "ac", "attached bathroom", "common bathroom")):
        return {"kb_key": "hostel_types", "content": KNOWLEDGE_BASE.get("hostel_types")}
    if any(word in q for word in ("fee", "fees", "payment", "deposit", "mess fee", "security deposit")):
        return {"kb_key": "fees", "content": KNOWLEDGE_BASE.get("fees")}
    if any(word in q for word in ("mess", "food", "breakfast", "lunch", "dinner", "menu")):
        return {"kb_key": "mess_info", "content": KNOWLEDGE_BASE.get("mess_info")}
    if any(word in q for word in ("rule", "timing", "time", "contact", "warden", "visit", "visiting", "laundry", "gate")):
        return {"kb_key": "faq", "content": KNOWLEDGE_BASE.get("faq")}
    return None

# -------------------------
# LLM helpers
# -------------------------
def _call_llm(prompt_text: str) -> str:
    try:
        resp = llm.invoke(prompt_text)
    except TypeError:
        try:
            resp = llm(prompt_text)
        except Exception as e:
            return f"LLM error: {e}"
    except Exception:
        try:
            resp = llm(prompt_text)
        except Exception as e:
            return f"LLM error: {e}"

    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return resp.content
    if hasattr(resp, "generations"):
        try:
            return resp.generations[0][0].text
        except Exception:
            pass
    if hasattr(resp, "text"):
        return resp.text
    return str(resp)

def ask_informational_llm(question: str, context: Optional[str] = None) -> Dict[str, Any]:
    prefix = (
        "You are a friendly and helpful hostel information assistant for parents and students.\n"
        "Answer concisely and politely. If the user explicitly requests live data (counts, availability, room occupancy), "
        "respond: 'I will check availability for you.' so the BI agent can be invoked.\n\n"
    )
    prompt = prefix + (context or "") + f"\nUser question: {question}\nAnswer briefly."
    text = _call_llm(prompt)
    return {"answer": text.strip(), "llm_raw": text}

# -------------------------
# Small utilities
# -------------------------
def tabular_preview(rows: List[Dict[str, Any]], max_rows: int = 8) -> str:
    """Return a small markdown-like table preview string for assistant answer."""
    if not rows:
        return "(no rows)"
    # limit rows
    sample = rows[:max_rows]
    # headers from first row
    headers = list(sample[0].keys())
    # build table string
    sep = " | "
    header_line = sep.join(headers)
    divider = " | ".join(["---"] * len(headers))
    body_lines = []
    for r in sample:
        row_vals = []
        for h in headers:
            v = r.get(h)
            # short string-safe representation
            s = str(v) if v is not None else ""
            if len(s) > 60:
                s = s[:57] + "..."
            row_vals.append(s.replace("\n", " "))
        body_lines.append(sep.join(row_vals))
    preview = header_line + "\n" + divider + "\n" + "\n".join(body_lines)
    if len(rows) > max_rows:
        preview += f"\n\n...and {len(rows)-max_rows} more rows."
    return preview

# -------------------------
# RAG helpers (compact)
# -------------------------
def build_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits):
        src = h.get("source", "<unknown>")
        txt = h.get("text", "")
        parts.append(f"[{i+1}] Source: {src}\n{txt}")
    return "\n\n".join(parts)

def answer_with_rag(question: str, hits: List[Dict[str, Any]], memory_context: Optional[str] = None, debug: bool = False) -> Dict[str, Any]:
    context = build_context_from_hits(hits)
    rag_prompt = f"""
You are a factual and careful hostel information assistant. Use ONLY the context below — DO NOT invent facts or use outside knowledge.
If the user's question asks for numbers or fees, return the exact values from the context. If the answer is not present, say: "I couldn't find that exact information in the knowledge base."

Memory context (if present):
{memory_context or "(none)"}

CONTEXT (top relevant passages):
{context}

User question: {question}

Answer in 1-3 short paragraphs. At the end include a line 'Sources: [1],[2]' referencing passage indices used.
"""
    llm_resp = _call_llm(rag_prompt)
    out = {
        "type": "informational",
        "answer": llm_resp.strip(),
        "llm_raw": llm_resp,
        "sources": [{"idx": i+1, "source": h.get("source"), "score": h.get("score")} for i, h in enumerate(hits)],
        "kb_hits": hits
    }
    if debug:
        out["rag_prompt"] = rag_prompt
    return out

# -------------------------
# Memory-fact helpers (quick follow-ups)
# -------------------------
def _extract_question_fact(question: str) -> dict:
    if not question:
        return {}
    q = question.lower()
    if re.search(r"\b(blocks?|block id|block number)\b", q):
        return {"fact_type": "block"}
    if re.search(r"\b(roll|rollno|roll no|roll number)\b", q):
        return {"fact_type": "rollno"}
    if re.search(r"\ballerg(?:y|ic|ic to)\b", q):
        return {"fact_type": "allergy"}
    return {}

def _answer_from_memory_fact(session_id: str, question: str, memory_hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qfact = _extract_question_fact(question)
    if not qfact or not session_id:
        return None
    needed = qfact.get("fact_type")
    for m in memory_hits:
        p = m.get("payload") or {}
        if p.get("fact_type") == needed and p.get("fact_value"):
            val = p.get("fact_value")
            ans = f"I have a note for this session: {needed} = {val}."
            return {"type": "informational", "answer": ans, "llm_raw": ans, "memory_used": memory_hits}
    # fallback regex scan
    if needed == "block":
        for m in memory_hits:
            text = (m.get("payload") or {}).get("text") or m.get("text") or ""
            if not text:
                continue
            mblk = re.search(r"\bblock[s]?\s*(?:no|id|number|#|is|:)?\s*([0-9]{1,4})\b", text.lower())
            if mblk:
                ans = f"I found in memory: block = {mblk.group(1)}."
                return {"type": "informational", "answer": ans, "llm_raw": ans, "memory_used": memory_hits}
    return None

# -------------------------
# Main orchestrator (exports function)
# -------------------------
def orchestrate_message(question: str, session_id: str = None, debug: bool = False) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"type": "informational", "answer": "Please provide a question.", "llm_raw": ""}

    # retrieve session memory
    memory_hits: List[Dict[str, Any]] = []
    if session_id:
        try:
            memory_hits = retrieve_memory(session_id, question, k=8) or []
        except Exception:
            memory_hits = []

    memory_context = ""
    if memory_hits:
        mem_lines = []
        for i, m in enumerate(memory_hits):
            p = m.get("payload") or {}
            txt = p.get("text") or p.get("summary") or ""
            ts = p.get("ts")
            mem_lines.append(f"[M{i+1}] {txt} (ts:{ts})")
        memory_context = "User memory (most relevant):\n" + "\n".join(mem_lines) + "\n\n"

    # quick memory-fact resolution
    mem_fact_answer = None
    try:
        mem_fact_answer = _answer_from_memory_fact(session_id, question, memory_hits) if session_id else None
    except Exception:
        mem_fact_answer = None
    if mem_fact_answer:
        if debug:
            mem_fact_answer["memory_used"] = memory_hits
            mem_fact_answer["memory_context"] = memory_context
        return mem_fact_answer

    # structured KB
    kb_struct = kb_lookup_struct(question)
    if kb_struct:
        kb_key = kb_struct.get("kb_key")
        content = kb_struct.get("content")
        # reuse small summarizer prompt
        prompt = f"Use ONLY the JSON below. Answer briefly.\n\n{json.dumps(content)[:5000]}\n\nQuestion: {question}"
        llm_text = _call_llm(prompt)
        if session_id:
            try:
                add_memory(session_id, f"user: {question}", {"role": "user"})
                add_memory(session_id, f"assistant: {llm_text}", {"role": "assistant", "kb_key": kb_key})
            except Exception:
                pass
        out = {"type": "informational", "answer": llm_text.strip(), "llm_raw": llm_text, "source": "knowledge_base"}
        if debug:
            out["memory_used"] = memory_hits
            out["memory_context"] = memory_context
        return out

    # BI -> NL->SQL
    if looks_like_data_question(question):
        bi_res = generate_sql_and_run(question)
        # handle errors or missing sql
        if bi_res.get("error"):
            err = bi_res.get("error")
            out = {"type": "bi", "answer": f"Error running query: {err}", "error": err, "sql": bi_res.get("sql")}
            if debug:
                out["memory_used"] = memory_hits
                out["memory_context"] = memory_context
            return out

        sql = bi_res.get("sql")
        data_rows = bi_res.get("data") or []
        safety_ok = bi_res.get("safety_ok", False)
        # create assistant-friendly textual preview so frontend shows result in assistant message
        preview = tabular_preview(data_rows, max_rows=8)
        rows_count = len(data_rows)
        answer_lines = [f"I ran a data query and found {rows_count} row(s)."]
        if sql:
            answer_lines.append("SQL:")
            # include SQL in fenced block for readability
            answer_lines.append(f"```sql\n{sql}\n```")
        answer_lines.append("Preview of results:")
        answer_lines.append(preview)
        assistant_answer = "\n\n".join(answer_lines)

        wrapped = {
            "type": "bi",
            "answer": assistant_answer,
            "prompt": bi_res.get("prompt"),
            "llm_raw": bi_res.get("llm_raw"),
            "sql": sql,
            "safety_ok": safety_ok,
            "error": bi_res.get("error"),
            "data": data_rows
        }

        # persist memory entries about the BI action
        if session_id:
            try:
                add_memory(session_id, f"user: {question}", {"role": "user"})
                add_memory(session_id, f"assistant: ran BI query: {sql}", {"role": "assistant", "type": "bi"})
            except Exception:
                pass

        if debug:
            wrapped["memory_used"] = memory_hits
            wrapped["memory_context"] = memory_context
        return wrapped

    # RAG retrieval
    try:
        hits = retrieve_top_k(question, k=6) or []
    except Exception:
        hits = []
    if hits:
        filtered = [h for h in hits if (h.get("score") is None or h.get("score") >= 0.0)]
        if filtered:
            rag_out = answer_with_rag(question, filtered, memory_context=memory_context, debug=debug)
            if session_id:
                try:
                    add_memory(session_id, f"user: {question}", {"role": "user"})
                    add_memory(session_id, f"assistant: {rag_out.get('answer')}", {"role": "assistant", "source": "rag"})
                except Exception:
                    pass
            if debug:
                rag_out["memory_used"] = memory_hits
                rag_out["memory_context"] = memory_context
            return rag_out

    # fallback informational LLM (include schema + memory context)
    info = ask_informational_llm(question, context=(memory_context or "") + (SCHEMA_TEXT or ""))
    if session_id:
        try:
            add_memory(session_id, f"user: {question}", {"role": "user"})
            add_memory(session_id, f"assistant: {info['answer']}", {"role": "assistant", "type": "informational"})
        except Exception:
            pass
    out = {"type": "informational", "answer": info["answer"], "llm_raw": info["llm_raw"]}
    if debug:
        out["memory_used"] = memory_hits
        out["memory_context"] = memory_context
    return out

# quick test when run directly
if __name__ == "__main__":
    import sys, pprint
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question> ")
    res = orchestrate_message(q, session_id="sid-test", debug=True)
    pprint.pprint(res)

