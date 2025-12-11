# chat_agent.py
"""
Chat orchestrator / router for Agentic HMS.

Responsibilities:
- Load a small knowledge base (KB) from backend/knowledge_base/*.json
- Decide whether an incoming user message should:
    * be answered from the KB (deterministic)
    * be answered by the informational LLM
    * be routed to the BI NL->SQL agent (generate_sql_and_run from nl_agent)
- Return a consistent dictionary describing the response so the frontend can render:
    - type: "informational" | "bi"
    - answer: assistant text (or KB json string)
    - llm_raw: raw model output (if applicable)
    - if bi: prompt, sql, safety_ok, error, data
"""
import os
import re
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# reuse existing BI agent
from nl_agent import generate_sql_and_run, SCHEMA_TEXT  # nl_agent must be importable from same folder

# LLM wrapper (match your project)
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# Initialize lightweight LLM for informational answers
# You can tune model & temperature here
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# Knowledge Base loading
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
                # ignore bad KB file but continue
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
    kw_hits = sum(1 for k in BI_KEYWORDS if k in t)
    pattern_hit = bool(PATTERN.search(text or ""))
    # heuristics: must have at least one BI keyword and one pattern hit OR explicit "how many" etc.
    if "how many" in t or "count" in t:
        return True
    return (kw_hits >= 1 and pattern_hit)

# -------------------------
# KB lookup
# -------------------------
def kb_lookup(question: str) -> Optional[str]:
    """
    Return a textual KB answer (string) if the question matches KB topics.
    We return JSON-stringified content for structured display; UI can format.
    """
    q = (question or "").lower()
    if not KNOWLEDGE_BASE:
        return None

    # hostel types
    if any(word in q for word in ("single", "double", "triple", "attached", "non ac", "ac", "attached bathroom", "common bathroom")):
        kb = KNOWLEDGE_BASE.get("hostel_types")
        return json.dumps(kb, indent=2) if kb else None

    # fees
    if any(word in q for word in ("fee", "fees", "payment", "deposit", "mess fee", "security deposit")):
        kb = KNOWLEDGE_BASE.get("fees")
        return json.dumps(kb, indent=2) if kb else None

    # mess information
    if any(word in q for word in ("mess", "food", "breakfast", "lunch", "dinner", "menu")):
        kb = KNOWLEDGE_BASE.get("mess_info")
        return json.dumps(kb, indent=2) if kb else None

    # general faq/rules/contact
    if any(word in q for word in ("rule", "timing", "time", "contact", "warden", "visit", "visiting", "laundry", "gate")):
        kb = KNOWLEDGE_BASE.get("faq")
        return json.dumps(kb, indent=2) if kb else None

    return None

# -------------------------
# LLM helpers
# -------------------------
def _call_llm(prompt_text: str) -> str:
    """
    Call ChatOpenAI and normalize response into a string.
    Handles a couple possible return shapes.
    """
    try:
        resp = llm.invoke(prompt_text)
    except TypeError:
        resp = llm(prompt_text)
    except Exception as e:
        # return the exception text to caller
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
# Main orchestrator
# -------------------------
def orchestrate_message(question: str) -> Dict[str, Any]:
    """
    Decide how to handle the incoming user message.

    Returns a dictionary with keys depending on the type:
    - type: "informational" or "bi"
    - answer: assistant text (or KB JSON)
    - llm_raw: raw LLM text if applicable
    - If type == "bi": includes prompt, sql, safety_ok, error, data (from generate_sql_and_run)
    """
    question = (question or "").strip()
    if not question:
        return {"type": "informational", "answer": "Please provide a question.", "llm_raw": ""}

    # 1) KB-first: deterministic lookup
    kb = kb_lookup(question)
    if kb:
        return {"type": "informational", "answer": kb, "source": "knowledge_base"}

    # 2) Heuristic: does this look like a BI data question?
    if looks_like_data_question(question):
        # call the BI NL->SQL agent
        bi_res = generate_sql_and_run(question)
        # if BI agent did not produce SQL or failed at generation, fallback to informational LLM
        if not bi_res.get("sql"):
            # fallback: ask LLM informationally with schema context
            info = ask_informational_llm(question, context=SCHEMA_TEXT)
            return {"type": "informational", "answer": info["answer"], "llm_raw": info["llm_raw"], "fallback": True}
        # success: return BI payload
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

    # 3) Default: informational LLM answer (include schema context to reduce hallucination)
    info = ask_informational_llm(question, context=SCHEMA_TEXT)
    return {"type": "informational", "answer": info["answer"], "llm_raw": info["llm_raw"]}

# If run standalone for quick testing
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question> ")
    print(json.dumps(orchestrate_message(q), indent=2, ensure_ascii=False))

