# kb_rag.py
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(ROOT, "../.env"))

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLL_NAME = os.environ.get("QDRANT_COLLECTION", "hostel_kb")

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings")
    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = []
    for t in texts:
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=t)
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

def chunk_text(text: str, max_chars: int = 1200):
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
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            for i in range(0, len(p), max_chars):
                out.append(p[i:i+max_chars])
    return out

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

def retrieve_top_k(query: str, k: int = 5, collection_name=COLL_NAME):
    vec = embed_texts([query])[0]
    # prefer qdrant.search if available
    try:
        if hasattr(qdrant, "search"):
            hits = qdrant.search(collection_name=collection_name, query_vector=vec, top=k)
            results = []
            for h in hits:
                results.append({"id": getattr(h, "id", None), "score": getattr(h, "score", None), "source": h.payload.get("source") if getattr(h, "payload", None) else None, "text": h.payload.get("text") if getattr(h, "payload", None) else None})
            return results
    except Exception:
        pass
    # fallback to HTTP API
    try:
        import requests
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/points/search"
        body = {"vector": vec, "top": k, "with_payload": True}
        r = requests.post(url, json=body, timeout=10)
        r.raise_for_status()
        j = r.json()
        hits = j.get("result") or j.get("hits") or []
        out = []
        for h in hits:
            payload = h.get("payload") or {}
            out.append({"id": h.get("id"), "score": h.get("score"), "source": payload.get("source"), "text": payload.get("text")})
        return out
    except Exception:
        return []

