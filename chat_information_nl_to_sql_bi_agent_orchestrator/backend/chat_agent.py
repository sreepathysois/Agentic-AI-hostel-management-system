# chat_agent.py
"""
Chat orchestrator / router for Agentic HMS (LLM-backed RAG + KB summarization).

Flow:
 - Structured KB lookup (local JSON files)
 - BI detection -> NL→SQL agent (generate_sql_and_run)
 - RAG retrieval (Qdrant) -> if hits found, ask LLM to answer using ONLY retrieved passages
 - Fallback: general informational LLM (with schema context)
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# reuse existing BI agent (flat file imports; container layout has files under /app)
from nl_agent import generate_sql_and_run, SCHEMA_TEXT

# retrieval helper (Qdrant)
from kb_rag import retrieve_top_k

# LLM wrapper (langchain_openai wrapper you use)
from langchain_openai import ChatOpenAI

# Load environment (if your .env is at /app/.env this still works)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# Initialize LLM (tune model/temperature as needed)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# Knowledge Base loading (local JSON)
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

# -------------------------
# Heuristics for intent
# -------------------------
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

# -------------------------
# KB lookup (structured)
# -------------------------
def kb_lookup_struct(question: str) -> Optional[Dict[str, Any]]:
    q = (question or "").lower()
    if not KNOWLEDGE_BASE:
        return None

    if any(word in q for word in ("single", "double", "triple", "attached", "non ac", "ac", "attached bathroom", "common bathroom")):
        kb = KNOWLEDGE_BASE.get("hostel_types")
        return {"kb_key": "hostel_types", "content": kb}

    if any(word in q for word in ("fee", "fees", "payment", "deposit", "mess fee", "security deposit")):
        kb = KNOWLEDGE_BASE.get("fees")
        return {"kb_key": "fees", "content": kb}

    if any(word in q for word in ("mess", "food", "breakfast", "lunch", "dinner", "menu")):
        kb = KNOWLEDGE_BASE.get("mess_info")
        return {"kb_key": "mess_info", "content": kb}

    if any(word in q for word in ("rule", "timing", "time", "contact", "warden", "visit", "visiting", "laundry", "gate")):
        kb = KNOWLEDGE_BASE.get("faq")
        return {"kb_key": "faq", "content": kb}

    return None

# -------------------------
# LLM helpers
# -------------------------
def _call_llm(prompt_text: str) -> str:
    """
    Call ChatOpenAI and normalize response into a string.
    """
    try:
        resp = llm.invoke(prompt_text)
    except TypeError:
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
    """
    Ask the LLM to answer a general informational question.
    We include SCHEMA_TEXT optionally for contextual accuracy.
    """
    prefix = (
        "You are a friendly and helpful hostel information assistant for parents and students.\n"
        "Answer concisely. If the user explicitly requests live data (counts, availability, room occupancy), "
        "respond: 'I will check availability for you.' so the BI agent can be invoked.\n\n"
    )
    prompt = prefix + (context or "") + f"\nUser question: {question}\nAnswer briefly."
    text = _call_llm(prompt)
    return {"answer": text.strip(), "llm_raw": text}

# -------------------------
# KB -> LLM summarization helpers (for structured KB)
# -------------------------
def _truncate_text(s: str, max_chars: int = 3000) -> str:
    if not s:
        return s
    return s if len(s) <= max_chars else s[:max_chars-100] + "\n\n...TRUNCATED..."

def build_kb_context(kb_key: str, content: dict, max_chars: int = 2500) -> str:
    if not content:
        return ""
    if kb_key == "fees":
        mini = {
            "hostel_fees": content.get("hostel_fees"),
            "mess_fees": content.get("mess_fees"),
            "security_deposit": content.get("security_deposit"),
            "notes": content.get("notes", []),
        }
        return _truncate_text(json.dumps(mini, ensure_ascii=False, indent=2), max_chars)
    if kb_key == "mess_info":
        mini = {
            "mess_timings": content.get("mess_timings"),
            "weekly_menu_sample": content.get("weekly_menu_sample")
        }
        return _truncate_text(json.dumps(mini, ensure_ascii=False, indent=2), max_chars)
    if kb_key == "hostel_types":
        mini = {"hostel_types": []}
        for t in content.get("hostel_types", []):
            mini["hostel_types"].append({
                "type": t.get("type"),
                "description": t.get("description"),
                "features": t.get("features"),
                "variants": t.get("variants")
            })
        return _truncate_text(json.dumps(mini, ensure_ascii=False, indent=2), max_chars)
    return _truncate_text(json.dumps(content, ensure_ascii=False, indent=2), max_chars)

def summarize_kb_with_llm(question: str, kb_key: str, content: dict) -> Dict[str, str]:
    kb_context = build_kb_context(kb_key, content)
    prompt = f"""
You are a helpful hostel information assistant for parents and students. Use ONLY the information provided below (do not invent facts). If the user's question asks for numbers or fees, return exact values from the provided KB. Keep the answer concise, friendly, and suitable for a parent reading on a website.

--- BEGIN KB CONTEXT (JSON) ---
{kb_context}
--- END KB CONTEXT ---

User question: {question}

Answer (1-3 short paragraphs). If the question is outside the KB, say: "I couldn't find that exact information in the knowledge base; would you like me to look it up?" Also include a final line "Source: KB".
"""
    llm_text = _call_llm(prompt)
    return {"answer": llm_text.strip(), "llm_raw": llm_text}

# -------------------------
# RAG helpers (using retrieve_top_k from kb_rag)
# -------------------------
def build_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    """
    Create a compact enumerated context text from top-k hits.
    """
    parts = []
    for i, h in enumerate(hits):
        src = h.get("source", "<unknown>")
        txt = h.get("text", "")
        parts.append(f"[{i+1}] Source: {src}\n{txt}")
    return "\n\n".join(parts)

def answer_with_rag(question: str, hits: List[Dict[str, Any]], debug: bool=False) -> Dict[str, Any]:
    """
    Ask the LLM to answer using ONLY the retrieved passages.
    Returns structured response compatible with frontend.
    """
    context = build_context_from_hits(hits)
    prompt = f"""
You are a factual and careful hostel information assistant. Use ONLY the context below — DO NOT invent facts or use outside knowledge.
If the user's question asks for numbers or fees, return the exact values from the context. If the answer is not present in the provided context, say: "I couldn't find that exact information in the knowledge base."

CONTEXT (top relevant passages):
{context}

User question: {question}

Answer in 1-3 short paragraphs. At the end, include a line 'Sources: [1],[2]' referencing passage indices used. Keep the answer concise and friendly.
"""
    llm_resp = _call_llm(prompt)
    return {
        "type": "informational",
        "answer": llm_resp.strip(),
        "llm_raw": llm_resp,
        "sources": [{"idx": i+1, "source": h.get("source"), "score": h.get("score")} for i, h in enumerate(hits)],
        "kb_hits": hits
    }
    if debug:
        out["rag_prompt"] = rag_prompt
        out["llm_prompt"] = rag_prompt  # here they are same; if you later wrap or modify, set llm_prompt accordingly
    return out

# -------------------------
# Main orchestrator
# -------------------------
def orchestrate_message(question: str, debug: bool=False) -> Dict[str, Any]:
    """
    Decide how to handle the incoming user message.

    Returns a dictionary with keys depending on the type:
    - type: "informational" or "bi"
    - answer: assistant text (human-friendly)
    - llm_raw: raw LLM text if applicable
    - If type == "bi": includes prompt, sql, safety_ok, error, data (from generate_sql_and_run)
    """
    question = (question or "").strip()
    if not question:
        return {"type": "informational", "answer": "Please provide a question.", "llm_raw": ""}

    # 1) Structured KB-first: if local KB JSON matches, summarize with LLM using KB context
    kb_struct = kb_lookup_struct(question)
    if kb_struct:
        kb_key = kb_struct.get("kb_key")
        content = kb_struct.get("content")
        summary = summarize_kb_with_llm(question, kb_key, content)
        return {
            "type": "informational",
            "answer": summary["answer"],
            "source": "knowledge_base",
            "kb_raw": content,
            "llm_raw": summary["llm_raw"]
        }

    # 2) Heuristic: does this look like a BI data question? If so, route to NL->SQL agent
    if looks_like_data_question(question):
        bi_res = generate_sql_and_run(question)
        if not bi_res.get("sql"):
            info = ask_informational_llm(question, context=SCHEMA_TEXT)
            return {"type": "informational", "answer": info["answer"], "llm_raw": info["llm_raw"], "fallback": True}
        rows = len(bi_res.get("data") or [])
        summary = f"I ran a data query and found {rows} row(s). See SQL and results."
        out = {
            "type": "bi",
            "answer": summary,
            "prompt": bi_res.get("prompt"),
            "llm_raw": bi_res.get("llm_raw"),
            "sql": bi_res.get("sql"),
            "safety_ok": bi_res.get("safety_ok"),
            "error": bi_res.get("error"),
            "data": bi_res.get("data")
        }
        return out

    # 3) RAG: try vector retrieval (Qdrant). If hits found, ask LLM to answer using only those passages.
    try:
        hits = retrieve_top_k(question, k=6)
    except Exception as e:
        hits = []
    if hits:
        # optional: filter very low-score hits if your kb_rag returns scores (skip if None)
        try:
            # simple score filter if numeric scores present
            filtered = [h for h in hits if (h.get("score") is None or h.get("score") >= 0.0)]
        except Exception:
            filtered = hits
        if filtered:
            return answer_with_rag(question, filtered, debug=debug)

    # 4) Default: general informational LLM answer (include schema context to reduce hallucination)
    info = ask_informational_llm(question, context=SCHEMA_TEXT)
    return {"type": "informational", "answer": info["answer"], "llm_raw": info["llm_raw"]}

# If run standalone for quick testing
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question> ")
    print(json.dumps(orchestrate_message(q), indent=2, ensure_ascii=False))

