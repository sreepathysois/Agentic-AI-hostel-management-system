# chat_agent.py
"""
Chat orchestrator / router for Agentic HMS (LLM-backed RAG + KB summarization + session memory).

This orchestrator:
 - loads structured KB JSONs (knowledge_base/*.json)
 - routes questions to:
    * structured KB summarizer (local JSON)
    * RAG (Qdrant via kb_rag.retrieve_top_k)
    * BI NL->SQL agent (nl_agent.generate_sql_and_run)
    * fallback informational LLM
 - uses:
    * Qdrant-based semantic memory (retrieve_memory, add_memory)
    * simple in-process chat history (add_chat_message, get_recent_chat_history)
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from nl_agent import generate_sql_and_run, SCHEMA_TEXT
from kb_rag import retrieve_top_k
from memory import (
    retrieve_memory,
    add_memory,
    add_chat_message,
    get_recent_chat_history,
    format_chat_history_for_prompt,
)

from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# KB loading
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
# BI detection heuristics
# -------------------------
BI_KEYWORDS = {
    "show", "list", "count", "total", "per", "by", "how many", "available",
    "vacant", "occupied", "seats", "rooms", "bookings", "pending", "fees",
    "payment", "availability", "seat", "block", "vacancy", "students",
}
PATTERN = re.compile(
    r"\b(block|seat|room|student|gender|fees|booking|date|month|year|vacant|available)\b",
    re.I,
)

def looks_like_data_question(text: str) -> bool:
    t = (text or "").lower()
    if "how many" in t or "count" in t:
        return True
    kw_hits = sum(1 for k in BI_KEYWORDS if k in t)
    pattern_hit = bool(PATTERN.search(text or ""))
    return kw_hits >= 1 and pattern_hit

# -------------------------
# KB structured lookup
# -------------------------
def kb_lookup_struct(question: str) -> Optional[Dict[str, Any]]:
    q = (question or "").lower()
    if not KNOWLEDGE_BASE:
        return None

    if any(
        word in q
        for word in (
            "single",
            "double",
            "triple",
            "attached",
            "non ac",
            "ac",
            "attached bathroom",
            "common bathroom",
        )
    ):
        return {"kb_key": "hostel_types", "content": KNOWLEDGE_BASE.get("hostel_types")}

    if any(
        word in q
        for word in ("fee", "fees", "payment", "deposit", "mess fee", "security deposit")
    ):
        return {"kb_key": "fees", "content": KNOWLEDGE_BASE.get("fees")}

    if any(word in q for word in ("mess", "food", "breakfast", "lunch", "dinner", "menu")):
        return {"kb_key": "mess_info", "content": KNOWLEDGE_BASE.get("mess_info")}

    if any(
        word in q
        for word in ("rule", "timing", "time", "contact", "warden", "visit", "visiting", "laundry", "gate")
    ):
        return {"kb_key": "faq", "content": KNOWLEDGE_BASE.get("faq")}

    return None

# -------------------------
# LLM helper
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
# RAG helpers
# -------------------------
def build_context_from_hits(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits):
        src = h.get("source", "<unknown>")
        txt = h.get("text", "")
        parts.append(f"[{i+1}] Source: {src}\n{txt}")
    return "\n\n".join(parts)

def answer_with_rag(
    question: str,
    hits: List[Dict[str, Any]],
    debug: bool = False,
    memory_context: Optional[str] = None,
) -> Dict[str, Any]:
    context = build_context_from_hits(hits)
    rag_prompt = f"""
You are a factual and careful hostel information assistant. Use ONLY the context below — DO NOT invent facts or use outside knowledge.
If the user's question asks for numbers or fees, return the exact values from the context. If the answer is not present, say: "I couldn't find that exact information in the knowledge base."

Memory context (if present):
{memory_context or "(none)"}

CONTEXT (top relevant passages):
{context}

User question: {question}

Answer in 1-3 short paragraphs. At the end, include a line 'Sources: [1],[2]' referencing passage indices used. Keep the answer concise and friendly.
"""
    llm_resp = _call_llm(rag_prompt)
    out = {
        "type": "informational",
        "answer": llm_resp.strip(),
        "llm_raw": llm_resp,
        "sources": [
            {
                "idx": i + 1,
                "source": h.get("source"),
                "score": h.get("score"),
            }
            for i, h in enumerate(hits)
        ],
        "kb_hits": hits,
    }
    if debug:
        out["rag_prompt"] = rag_prompt
    return out

# -------------------------
# Memory-fact helpers (for quick follow-ups)
# -------------------------
def _extract_question_fact(question: str) -> dict:
    if not question:
        return {}
    q = question.lower()
    if re.search(r"\b(my|my interested|the block i)\b", q) and re.search(
        r"\b(block|block id|block number)\b", q
    ):
        return {"fact_type": "block"}
    if re.search(r"\b(which|what).*(block|block id|block number)\b", q):
        return {"fact_type": "block"}
    if re.search(r"\b(roll|rollno|roll no|roll number)\b", q):
        return {"fact_type": "rollno"}
    if re.search(r"\ballerg(?:y|ic|ic to)\b", q):
        return {"fact_type": "allergy"}
    return {}

def _answer_from_memory_fact(
    session_id: str, question: str, memory_hits: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    qfact = _extract_question_fact(question)
    if not qfact or not session_id:
        return None
    needed = qfact.get("fact_type")

    # direct fact entries from payload
    for m in memory_hits:
        p = m.get("payload") or {}
        if p.get("fact_type") == needed and p.get("fact_value"):
            val = p.get("fact_value")
            ans = f"I have a note for this session: {needed} = {val}."
            return {
                "type": "informational",
                "answer": ans,
                "llm_raw": ans,
                "memory_used": memory_hits,
            }

    # fallback regex scan in memory text
    if needed == "block":
        for m in memory_hits:
            text = (m.get("payload") or {}).get("text") or m.get("text") or ""
            if not text:
                continue
            mblk = re.search(
                r"\bblock[s]?\s*(?:no|id|number|#|is|:)?\s*([0-9]{1,4})\b",
                text.lower(),
            )
            if mblk:
                ans = f"I found in memory: block = {mblk.group(1)}."
                return {
                    "type": "informational",
                    "answer": ans,
                    "llm_raw": ans,
                    "memory_used": memory_hits,
                }

    return None

# -------------------------
# Structured memory builder
# -------------------------
def build_structured_memory_from_hits(
    memory_hits: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Collapse memory_hits into a small JSON-like dict with stable session facts.
    This is fed into the NL->SQL agent as 'memory_context'.
    """
    state = {
        "session_block": None,
        "session_rollno": None,
        "session_allergies": [],
    }

    for m in memory_hits or []:
        p = m.get("payload") or {}
        ft = p.get("fact_type")
        fv = p.get("fact_value")
        if not fv:
            continue

        if ft == "block" and not state["session_block"]:
            state["session_block"] = fv

        elif ft == "rollno" and not state["session_rollno"]:
            state["session_rollno"] = fv

        elif ft == "allergy":
            if fv not in state["session_allergies"]:
                state["session_allergies"].append(fv)

    return state

# -------------------------
# Main orchestrator
# -------------------------
def orchestrate_message(
    question: str, session_id: str = None, debug: bool = False
) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {
            "type": "informational",
            "answer": "Please provide a question.",
            "llm_raw": "",
        }

    # 0a) recent chat history (like ConversationBufferMemory)
    history = []
    history_text = ""
    if session_id:
        try:
            history = get_recent_chat_history(session_id, limit=10)
            history_text = format_chat_history_for_prompt(history)
        except Exception:
            history = []
            history_text = ""

    # 0b) retrieve semantic memory from Qdrant
    memory_hits: List[Dict[str, Any]] = []
    if session_id:
        try:
            memory_hits = retrieve_memory(session_id, question, k=8) or []
        except Exception:
            memory_hits = []

    # build memory_context string for debug (shown in DebugPanel)
    memory_context = ""
    if memory_hits:
        mem_lines = []
        for i, m in enumerate(memory_hits):
            p = m.get("payload") or {}
            txt = p.get("text") or p.get("summary") or ""
            ts = p.get("ts")
            mem_lines.append(f"[M{i+1}] {txt} (ts:{ts})")
        memory_context = (
            "User memory (most relevant):\n" + "\n".join(mem_lines) + "\n\n"
        )

    # build structured JSON memory for agents (facts + chat history)
    structured_memory = build_structured_memory_from_hits(memory_hits)
    if history_text:
        structured_memory["conversation_history"] = history_text
    structured_memory_json = json.dumps(structured_memory, ensure_ascii=False)

    # quick memory-fact resolution
    try:
        mem_fact_answer = (
            _answer_from_memory_fact(session_id, question, memory_hits)
            if session_id
            else None
        )
    except Exception:
        mem_fact_answer = None

    if mem_fact_answer:
        # log chat history
        if session_id:
            try:
                add_chat_message(session_id, "user", question)
                add_chat_message(session_id, "assistant", mem_fact_answer["answer"])
            except Exception:
                pass
        if debug:
            mem_fact_answer["memory_used"] = memory_hits
            mem_fact_answer["memory_context"] = memory_context
        return mem_fact_answer

    # 1) structured KB lookup
    kb_struct = kb_lookup_struct(question)
    if kb_struct:
        kb_key = kb_struct.get("kb_key")
        content = kb_struct.get("content")
        prompt = (
            f"Use ONLY the JSON below. Answer briefly.\n\n"
            f"{json.dumps(content)[:5000]}\n\nQuestion: {question}"
        )
        llm_text = _call_llm(prompt)
        answer_text = llm_text.strip()

        # persist semantic + chat memory
        if session_id:
            try:
                add_memory(session_id, f"user: {question}", {"role": "user"})
                add_memory(
                    session_id,
                    f"assistant: {answer_text}",
                    {"role": "assistant", "kb_key": kb_key},
                )
                add_chat_message(session_id, "user", question)
                add_chat_message(session_id, "assistant", answer_text)
            except Exception:
                pass

        out = {
            "type": "informational",
            "answer": answer_text,
            "llm_raw": llm_text,
            "source": "knowledge_base",
        }
        if debug:
            out["memory_used"] = memory_hits
            out["memory_context"] = memory_context
        return out

    # 2) BI detection -> NL->SQL agent (uses structured_memory_json)
    if looks_like_data_question(question):
        bi_res = generate_sql_and_run(
            question,
            memory_context=structured_memory_json,
        )

        if bi_res.get("error"):
            answer = f"Error running query: {bi_res.get('error')}"
        elif not bi_res.get("sql"):
            # fallback informational LLM
            info = ask_informational_llm(question, context=SCHEMA_TEXT)
            answer = info["answer"]
            res = {
                "type": "informational",
                "answer": answer,
                "llm_raw": info["llm_raw"],
                "fallback": True,
            }
            if session_id:
                try:
                    add_memory(session_id, f"user: {question}", {"role": "user"})
                    add_memory(
                        session_id,
                        f"assistant: {answer}",
                        {"role": "assistant", "type": "informational"},
                    )
                    add_chat_message(session_id, "user", question)
                    add_chat_message(session_id, "assistant", answer)
                except Exception:
                    pass
            if debug:
                res["memory_used"] = memory_hits
                res["memory_context"] = memory_context
            return res
        else:
            rows = len(bi_res.get("data") or [])
            answer = (
                f"I ran a data query and found {rows} row(s). "
                f"See SQL and results below."
            )

        wrapped = {
            "type": "bi",
            "answer": answer,
            "prompt": bi_res.get("prompt"),
            "llm_raw": bi_res.get("llm_raw"),
            "sql": bi_res.get("sql"),
            "safety_ok": bi_res.get("safety_ok"),
            "error": bi_res.get("error"),
            "data": bi_res.get("data"),
        }

        if session_id:
            try:
                add_memory(session_id, f"user: {question}", {"role": "user"})
                add_memory(
                    session_id,
                    f"assistant: ran BI query: {bi_res.get('sql')}",
                    {"role": "assistant", "type": "bi"},
                )
                add_chat_message(session_id, "user", question)
                add_chat_message(session_id, "assistant", answer)
            except Exception:
                pass

        if debug:
            wrapped["memory_used"] = memory_hits
            wrapped["memory_context"] = memory_context
        return wrapped

    # 3) RAG retrieval
    hits = []
    try:
        hits = retrieve_top_k(question, k=6) or []
    except Exception:
        hits = []
    if hits:
        filtered = [
            h for h in hits if (h.get("score") is None or h.get("score") >= 0.0)
        ]
        if filtered:
            rag_out = answer_with_rag(
                question,
                filtered,
                debug=debug,
                memory_context=memory_context,
            )
            answer_text = rag_out.get("answer", "")
            if session_id:
                try:
                    add_memory(session_id, f"user: {question}", {"role": "user"})
                    add_memory(
                        session_id,
                        f"assistant: {answer_text}",
                        {"role": "assistant", "source": "rag"},
                    )
                    add_chat_message(session_id, "user", question)
                    add_chat_message(session_id, "assistant", answer_text)
                except Exception:
                    pass
            if debug:
                rag_out["memory_used"] = memory_hits
                rag_out["memory_context"] = memory_context
            return rag_out

    # 4) Fallback: informational LLM (include schema + memory_context)
    info = ask_informational_llm(
        question,
        context=(memory_context or "") + (SCHEMA_TEXT or ""),
    )
    answer_text = info["answer"]
    if session_id:
        try:
            add_memory(session_id, f"user: {question}", {"role": "user"})
            add_memory(
                session_id,
                f"assistant: {answer_text}",
                {"role": "assistant", "type": "informational"},
            )
            add_chat_message(session_id, "user", question)
            add_chat_message(session_id, "assistant", answer_text)
        except Exception:
            pass

    out = {
        "type": "informational",
        "answer": answer_text,
        "llm_raw": info["llm_raw"],
    }
    if debug:
        out["memory_used"] = memory_hits
        out["memory_context"] = memory_context
    return out

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question> ")
    print(
        json.dumps(
            orchestrate_message(q, session_id="sid-test", debug=True),
            indent=2,
            ensure_ascii=False,
        )
    )

