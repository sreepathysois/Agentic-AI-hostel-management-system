# backend/kb_rag.py
"""
KB RAG: ingest knowledge_base JSON files into Qdrant using embeddings,
and provide retrieval helper to return top-k relevant passages.

Default: uses OpenAI embeddings (text-embedding-3-small). Set EMBED_PROVIDER=local to use sentence-transformers.
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
import requests

ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(ROOT, "../.env"))

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")  # "openai" or "local"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLL_NAME = os.environ.get("QDRANT_COLLECTION", "hostel_kb")

# Connect to Qdrant
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ---------------- Embedding functions ----------------

def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings")

    client = OpenAI(api_key=OPENAI_API_KEY)

    embeddings = []
    for t in texts:
        resp = client.embeddings.create(
            model=OPENAI_EMBED_MODEL,   # default: text-embedding-3-small
            input=t
        )
        emb = resp.data[0].embedding
        embeddings.append(emb)

    return embeddings

def embed_texts_local(texts: List[str]) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=False)
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

def embed_texts(texts: List[str]) -> List[List[float]]:
    if EMBED_PROVIDER == "local":
        return embed_texts_local(texts)
    return embed_texts_openai(texts)

# ---------------- Qdrant helpers ----------------
def create_collection_if_not_exists(collection_name=COLL_NAME, vector_size=1536):
    try:
        existing = qdrant.get_collection(collection_name, ignore_missing=True)
    except Exception:
        existing = None
    if existing:
        return
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts = []
    cur = ""
    for line in text.splitlines():
        if len(cur) + len(line) + 1 > max_chars:
            parts.append(cur.strip())
            cur = line
        else:
            cur += ("\n" + line if cur else line)
    if cur:
        parts.append(cur.strip())
    # hard-split if still too long
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            for i in range(0, len(p), max_chars):
                out.append(p[i:i+max_chars])
    return out

# ---------------- Ingest ----------------
def ingest_kb_folder(kb_folder: str = os.path.join(ROOT, "knowledge_base"), collection_name=COLL_NAME):
    docs = []
    if not os.path.isdir(kb_folder):
        print("KB folder not found:", kb_folder)
        return
    for fname in sorted(os.listdir(kb_folder)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(kb_folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print("Failed to load", path, e)
                continue

        # Convert JSON to readable passages (simple extraction)
        def json_to_passages(obj, prefix=""):
            passages = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}{k}"
                    if isinstance(v, (dict, list)):
                        passages += json_to_passages(v, prefix=key + " > ")
                    else:
                        passages.append(f"{key}: {v}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:200]):
                    passages += json_to_passages(item, prefix=prefix + f"{i+1}. ")
            else:
                passages.append(f"{prefix}: {str(obj)}")
            return passages

        passages = json_to_passages(data)
        for p in passages:
            for chunk in chunk_text(p):
                docs.append({"source": fname, "text": chunk})

    if not docs:
        print("No passages found to ingest.")
        return

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} passages (provider={EMBED_PROVIDER})...")
    vectors = embed_texts(texts)
    if not vectors:
        raise RuntimeError("No vectors generated")

    vector_size = len(vectors[0])
    create_collection_if_not_exists(collection_name, vector_size=vector_size)

    points = []
    for i, (vec, d) in enumerate(zip(vectors, docs)):
        points.append({"id": i, "vector": vec, "payload": {"source": d["source"], "text": d["text"]}})

    BATCH = 128
    print("Upserting into Qdrant...")
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        qdrant.upsert(collection_name=collection_name, points=batch)
    print("Ingest complete.")

# ---------------- Retrieve ----------------


def _fetch_point_payload_http(point_id: int, collection_name=COLL_NAME) -> Dict[str, Any]:
    """Fetch a single point's payload via Qdrant HTTP GET (returns dict or {})."""
    try:
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/points/{point_id}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        j = r.json()
        # Qdrant GET returns {"result": {"id":..., "payload": {...}, "vector":[...]} }
        res = j.get("result") or j
        payload = res.get("payload") or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}

def retrieve_top_k(query: str, k: int = 5, collection_name=COLL_NAME) -> List[Dict[str, Any]]:
    """
    Robust retrieval: run vector search (client or HTTP), then ensure each hit has payload.
    If payload missing, attempt HTTP point GET to fetch payload.
    Returns list of {"id","score","source","text"}.
    """
    vec = embed_texts([query])[0]

    # normalization helper
    def _normalize_single(raw_item):
        # raw_item may be object-like or dict-like
        hid = getattr(raw_item, "id", None) if not isinstance(raw_item, dict) else raw_item.get("id")
        score = getattr(raw_item, "score", None) if not isinstance(raw_item, dict) else raw_item.get("score")
        # try payload locations
        if not isinstance(raw_item, dict):
            payload = getattr(raw_item, "payload", None) or {}
        else:
            payload = raw_item.get("payload") or raw_item.get("payload", {})
        # ensure payload is dict
        if payload is None:
            payload = {}

        # try to extract source/text from payload
        src = None
        text = None
        if isinstance(payload, dict):
            # common keys
            src = payload.get("source") or payload.get("src") or payload.get("file") or payload.get("filename")
            text = payload.get("text") or payload.get("content") or payload.get("body") or payload.get("payload")
            # if the payload contains nested result (some responses), check that
            if text is None and "result" in payload and isinstance(payload["result"], dict):
                nested = payload["result"]
                text = nested.get("text") or nested.get("content") or nested.get("payload")
                src = src or nested.get("source")
        return {"id": hid, "score": score, "source": src, "text": text}

    # 1) Try qdrant-client python search variants first
    raw_hits = []
    try:
        if hasattr(qdrant, "search"):
            raw_hits = qdrant.search(collection_name=collection_name, query_vector=vec, top=k, with_payload=True)
        elif hasattr(qdrant, "search_points"):
            raw_hits = qdrant.search_points(collection_name=collection_name, query_vector=vec, top=k, with_payload=True)
    except TypeError:
        # some clients don't accept named param; try without and let fallback handle payload fetch
        try:
            if hasattr(qdrant, "search"):
                raw_hits = qdrant.search(collection_name=collection_name, query_vector=vec, top=k)
            elif hasattr(qdrant, "search_points"):
                raw_hits = qdrant.search_points(collection_name=collection_name, query_vector=vec, top=k)
        except Exception:
            raw_hits = []
    except Exception:
        raw_hits = []

    # 2) If python client search yielded nothing, try HTTP search
    if not raw_hits:
        try:
            url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/points/search"
            body = {"vector": vec, "top": k, "with_payload": True}
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
            j = resp.json()
            # Qdrant HTTP returns under "result" (or "result"/"hits")
            raw_hits = j.get("result") or j.get("hits") or []
        except Exception:
            raw_hits = []

    # 3) Normalize and ensure payload present (if missing, GET the point)
    out = []
    for item in raw_hits:
        norm = _normalize_single(item)
        hid = norm.get("id")
        src = norm.get("source")
        text = norm.get("text")

        # If payload fields missing, fetch the point via HTTP GET (robust)
        if (src is None or text is None) and hid is not None:
            payload = _fetch_point_payload_http(hid, collection_name=collection_name)
            if isinstance(payload, dict):
                src = src or payload.get("source") or payload.get("src") or payload.get("file") or payload.get("filename")
                text = text or payload.get("text") or payload.get("content") or payload.get("body") or payload.get("payload")

        out.append({"id": hid, "score": norm.get("score"), "source": src, "text": text})

    return out

# CLI convenience
if __name__ == "__main__":
    print("Ingesting KB folder...")
    ingest_kb_folder(kb_folder=os.path.join(ROOT, "knowledge_base"))
    print("Done. Try retrieve_top_k('mess timings', 5)")

