# memory.py
import os
import time
import hashlib
import requests
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from kb_rag import embed_texts, qdrant, QDRANT_HOST, QDRANT_PORT, COLL_NAME
from qdrant_client.http.models import VectorParams, Distance

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

USER_MEM_COLL = os.environ.get("QDRANT_USER_MEMORY_COLLECTION", "user_memory")
DEFAULT_VEC_SIZE = int(os.environ.get("EMBED_DIM", "1536"))

def create_user_memory_collection_if_missing(vector_size: int = DEFAULT_VEC_SIZE):
    try:
        existing = qdrant.get_collection(USER_MEM_COLL, ignore_missing=True)
        if existing:
            return
    except Exception:
        existing = None
    qdrant.recreate_collection(
        collection_name=USER_MEM_COLL,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def _make_id(session_id: str, text: str, ts: int) -> int:
    seed = f"{session_id}:{ts}:{text}"[:4000]
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:15]
    return int(h, 16)

def _extract_simple_facts(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {}
    txt = text.lower()
    m = re.search(r"\bblock[s]?\s*(?:no|id|number|#|is|:)?\s*([0-9]{1,4})\b", txt)
    if m:
        return {"fact_type": "block", "fact_value": m.group(1)}
    m2 = re.search(r"\broll(?:\s*no|no)?\s*(?:[:#]|\s)?\s*([A-Za-z0-9\-]+)\b", txt)
    if m2:
        return {"fact_type": "rollno", "fact_value": m2.group(1)}
    m3 = re.search(r"\ballerg(?:y|ic|ic to)\s*(?:to)?\s*([a-zA-Z0-9 ,&-]+)", txt)
    if m3:
        return {"fact_type": "allergy", "fact_value": m3.group(1).strip()}
    return {}

def add_memory(session_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
    if not session_id or not text:
        raise ValueError("session_id and text required to add memory")
    create_user_memory_collection_if_missing()
    emb = embed_texts([text])[0]
    ts = int(time.time())
    pid = _make_id(session_id, text, ts)
    facts = _extract_simple_facts(text)
    payload = {"session_id": session_id, "text": text, "meta": meta or {}, "ts": ts}
    if facts:
        payload.update(facts)
    qdrant.upsert(collection_name=USER_MEM_COLL, points=[{"id": pid, "vector": emb, "payload": payload}])
    return pid

def retrieve_memory(session_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    if not session_id:
        return []
    vec = embed_texts([query])[0]
    try:
        filter_obj = {"must": [{"key": "session_id", "match": {"value": session_id}}]}
        if hasattr(qdrant, "search"):
            raw = qdrant.search(collection_name=USER_MEM_COLL, query_vector=vec, top=k, with_payload=True, filter=filter_obj)
        else:
            raw = qdrant.search_points(collection_name=USER_MEM_COLL, query_vector=vec, top=k, with_payload=True, filter=filter_obj)
        out = []
        for h in raw:
            payload = getattr(h, "payload", None) or {}
            out.append({"id": getattr(h, "id", None), "score": getattr(h, "score", None), "payload": payload})
        return out
    except Exception:
        try:
            url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{USER_MEM_COLL}/points/search"
            body = {"vector": vec, "top": k, "with_payload": True, "filter": {"must": [{"key": "session_id", "match": {"value": session_id}}]}}
            r = requests.post(url, json=body, timeout=10)
            r.raise_for_status()
            j = r.json()
            hits = j.get("result") or j.get("hits") or []
            out = []
            for h in hits:
                payload = h.get("payload") or {}
                out.append({"id": h.get("id"), "score": h.get("score"), "payload": payload})
            return out
        except Exception:
            return []

